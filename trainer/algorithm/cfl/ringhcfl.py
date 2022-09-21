from copy import deepcopy

import numpy as np
import torch
from frozenlist import FrozenList
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter

from trainer.algorithm.cfl.ringcfl import RingCFL
from trainer.core.aggregator import StateAggregator
from utils.nn.functional import state2vector


class RingHCFL(RingCFL):

    def _parse_kwargs(self, **kwargs):
        super(RingHCFL, self)._parse_kwargs(**kwargs)
        if hc := kwargs['hc']:
            # self.group_num = hc.get('group_num', 2)
            self.clustering = hc.get('clustering', 50)

    def _schedule_group(self, cids):
        cids = FrozenList(self._fds)
        grads = [grad for cid in cids if (grad := self._cache.get(cid, None))]
        if len(grads) == len(cids) > 1 and self._k > self.clustering:
            self.clustering = self.round
            M = cosine_similarity(torch.stack(state2vector(grads)).cpu().detach().numpy())
            M_bak = np.copy(M)
            M_bak[np.where(M_bak < 0.) or np.where(M_bak > 1.)] = np.NAN
            _, labels = cluster.affinity_propagation(M, damping=0.8)
            for gid in frozenset(labels):
                self._groups[gid] = {
                    'state': deepcopy(self._model.state_dict()),
                    'clients': set(c for c, l in zip(cids, labels) if l == gid)
                }
                self.writers[gid] = SummaryWriter(f'{self.writer.log_dir}/{gid}')
                self._aggregators[gid] = StateAggregator()
            self.rho = min(1. / len(self._groups), self.rho)