from collections import Counter
from copy import deepcopy
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
from trainer.algorithm.cluster.base import ClusteredFL, grouping
from trainer.utils.cluster import agglomerative_clustering
from utils.nn.functional import state2vector


class CFL(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(CFL, self)._parse_kwargs(**kwargs)
        if cfl := kwargs['cfl']:
            self.eps_1 = cfl.get('eps_1', 0.026)
            self.eps_2 = cfl.get('eps_2', 0.4)

    def _init_group(self):
        return {
            0: {'state': deepcopy(self._model.state_dict()), 'clients': set(self._fds)}
        }

    def _local_update_args(self, cids):
        self._flush_group()
        return super(CFL, self)._local_update_args(cids)

    def _local_update_hook(self, cid, res):
        self._cache[cid] = res[0]
        super(CFL, self)._local_update_hook(cid, res)

    def _flush_group(self):
        for gid in frozenset(self._groups.keys()):
            cids = self._groups[gid]['clients']
            res = self._cluster(cids)
            if res is not None:
                new_gid = len(self._groups)
                self._groups[new_gid] = {
                    'state': deepcopy(self._groups[gid]['state']),
                    'clients': set(res[1]),
                    'writer': SummaryWriter(f'{self._writer.log_dir}/{new_gid}'),
                    'aggregator': self._build_group_aggregator()
                }
                self._groups[gid]['clients'] -= self._groups[new_gid]['clients']
                for cid in cids:
                    self._cache.delete(cid)

    def _cluster(self, cids):
        grads = [grad for cid in cids if (grad := self._cache.get(cid, None))]
        if len(grads) < len(cids):
            return
        vecs = state2vector(grads)
        max_norm = torch.max(torch.stack([torch.norm(v) for v in vecs])).item()
        mean_norm = torch.norm(torch.mean(torch.stack(vecs), dim=0)).item()
        self._logger.info(f"max_norm: {round(max_norm, 3)}  mean_norm: {round(mean_norm, 3)}")
        if mean_norm > self.eps_1 or max_norm < self.eps_2:
            return
        M = - cosine_similarity(torch.stack(vecs).detach().numpy())
        labels = agglomerative_clustering(M)
        self._logger.info(f'Cluster result: {Counter(labels)}')
        return grouping(cids, labels)
