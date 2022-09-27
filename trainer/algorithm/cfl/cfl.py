from copy import deepcopy

import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter

from trainer.core.aggregator import StateAggregator
from trainer.core.proto import ClusteredFL
from utils.nn.functional import state2vector


class CFL(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(CFL, self)._parse_kwargs(**kwargs)
        if cfl := kwargs['cfl']:
            self.eps_1 = cfl.get('eps_1', 0.026)
            self.eps_2 = cfl.get('eps_2', 0.4)

    def _init_group_hook(self):
        self._groups[0] = {
            'state': deepcopy(self._model.state_dict()),
            'clients': set(self._fds)
        }

    def _schedule_group(self, cids):
        for gid in frozenset(self._groups.keys()):
            self._split_group(gid)

    def _split_group(self, gid):
        cids = self._groups[gid]['clients']
        grads = [grad for cid in cids if (grad := self._cache.get(cid, None))]
        if len(grads) == len(cids) > 1:
            vecs = state2vector(grads)
            max_norm = torch.max(torch.stack([torch.norm(v) for v in vecs])).item()
            mean_norm = torch.norm(torch.mean(torch.stack(vecs), dim=0)).item()
            self._print_msg("{}: max_norm: {:.3f}\tmean_norm: {:.3f}".format(gid, max_norm, mean_norm))
            if mean_norm < self.eps_1 and max_norm > self.eps_2:
                M = - cosine_similarity(torch.stack(vecs).detach().numpy())
                res = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(M)
                cids_ = [cid for cid, label in zip(cids, res.labels_) if label == 1]
                new_id = len(self._groups)
                self._groups[new_id] = {'state': deepcopy(self._groups[gid]['state']), 'clients': set(cids_)}
                self.writers[new_id] = SummaryWriter(f'{self.writer.log_dir}/{new_id}')
                self._aggregators[new_id] = StateAggregator()
                for cid in cids:
                    self._cache.delete(cid)
                self._groups[gid]['clients'] -= set(cids_)

    def _local_update_hook(self, cid, res):
        super(CFL, self)._local_update_hook(cid, res)
        self._cache[cid] = res[0]
