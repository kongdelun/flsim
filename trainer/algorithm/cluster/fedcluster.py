from collections import Counter
from copy import deepcopy
from functools import reduce
from operator import concat

from sklearn.metrics.pairwise import cosine_similarity
from trainer.algorithm.cluster.base import ClusteredFL, grouping
from trainer.utils.cluster import k_means
from utils.nn.functional import flatten
from utils.nn.stats import to_numpy
from utils.select import random_select


class FedCluster(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedCluster, self)._parse_kwargs(**kwargs)
        if cluster := kwargs['cluster']:
            self.group_num = cluster.get('group_num', 5)
            self.temp = cluster.get('temp', 1.)
        self.pretrain_state = deepcopy(self._model.state_dict())

    def _init_group(self):
        res, groups = self._cluster(list(self._fds)), {}
        for gid in res:
            groups[gid] = {
                'state': deepcopy(self.pretrain_state),
                'clients': set(res[gid])
            }
        return groups

    def _cluster(self, cids):
        self._pretrain(cids)
        X = to_numpy(list(map(lambda cid: flatten(self._cache[cid]['grad']), cids)))
        M = cosine_similarity(X)
        labels = k_means(
            M, self.group_num, self.seed,
            max_size=int(len(cids) / self.group_num),
            min_size=int(len(cids) / self.group_num)
        )
        self._logger.info(f'Cluster result: {Counter(labels)}')
        return grouping(cids, labels)

    def _pretrain(self, cids):
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self.pretrain_state, self._fds.train(c), self.local_args)
            for c in cids
        ])):
            self._cache[cid] = {
                'grad': res[0]
            }

    def _select_client(self):
        return reduce(concat, [
            random_select(
                list(self._groups[gid]['clients']),
                s_alpha=self.sample_rate,
                seed=self.seed + self._k
            ) for gid in self._groups
        ])
