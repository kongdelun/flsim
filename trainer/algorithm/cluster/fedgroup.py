from collections import Counter
from copy import deepcopy
import torch
from trainer.algorithm.cluster.base import ClusteredFL, grouping
from trainer.utils.cluster import agglomerative_clustering
from trainer.utils.matrix import madc
from utils.nn import stats
from utils.nn.aggregate import average
from utils.nn.functional import flatten, add
from utils.nn.stats import to_numpy
from utils.select import random_select


class FedGroup(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedGroup, self)._parse_kwargs(**kwargs)
        if group := kwargs['group']:
            self.group_num = group.get('group_num', 2)
            self.init_args = group.get('init_args', {
                'sample_rate': 0.4,
                'epoch': 10
            })
            self.pretrain_args = {
                'opt': self.opt,
                'batch_size': self.batch_size,
                'epoch': self.init_args['epoch']
            }
            self.pretrain_state = deepcopy(self._model.state_dict())

    def _init_group(self):
        self._logger.info("Grouping ......")
        cids = random_select(self._fds, s_alpha=self.init_args['sample_rate'])
        res, groups = self._cluster(cids), {}
        for gid in res:
            grad = average(
                [self._cache[c]['grad'] for c in res[gid]],
                [self._cache[c]['num_sample'] for c in res[gid]]
            )
            groups[gid] = {
                'num_sample': sum([self._cache[c]['num_sample'] for c in res[gid]]),
                'grad': grad,
                'state': add(self.pretrain_state, grad),
                'clients': set(res[gid])
            }
        return groups

    def _cluster(self, cids):
        self._pretrain(cids)
        X = to_numpy(list(map(lambda cid: flatten(self._cache[cid]['grad']), cids)))
        # M = edc(X, self.group_num, self.seed)
        M = madc(X)
        # labels = k_means(M, self.group_num, self.seed)
        labels = agglomerative_clustering(M)
        self._logger.info(f'Cluster result: {Counter(labels)}')
        return grouping(cids, labels)

    def _local_update_args(self, cids):
        self._flush_group(cids)
        return super(FedGroup, self)._local_update_args(cids)

    def _pretrain(self, cids):
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self.pretrain_state, self._fds.train(c), self.pretrain_args)
            for c in cids
        ])):
            self._cache[cid] = {
                'grad': res[0],
                'num_sample': res[1][0]
            }

    def _best_group(self, cid):
        idx = torch.argmin(torch.tensor([
            stats.cosine_dissimilarity(flatten(self._cache[cid]['grad']), flatten(self._groups[gid]['grad'])).item()
            for gid in self._groups
        ])).item()
        return list(self._groups.keys())[idx]

    def _flush_group(self, cids):
        new_cids = list(filter(lambda x: self._gid(x) is None, cids))
        self._pretrain(new_cids)
        for cid in new_cids:
            gid = self._best_group(cid)
            self._groups[gid]['grad'] = average(
                [self._groups[gid]['grad'], self._cache[cid]['grad']],
                [self._groups[gid]['num_sample'], self._cache[cid]['num_sample']]
            )
            self._groups[gid]['clients'].add(cid)
        self._logger.info("[{}] New: {:.1%}  Change: {:.1%}".format(self._k, len(new_cids) / len(cids), 0.))
