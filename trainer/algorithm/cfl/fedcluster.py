from copy import deepcopy
from functools import reduce
from operator import concat

import numpy as np
import torch
import torch.nn.functional as F
from frozenlist import FrozenList
from k_means_constrained import KMeansConstrained
from ray.util import ActorPool
from sklearn import cluster
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from trainer.algorithm.fedprox import ProxActor
from trainer.core.proto import ClusteredFL
from utils.compressor.basic import TopkCompressor
from utils.nn.aggregate import average
from utils.nn.functional import zero_like, linear_sum, add_, add
from utils.select import random_select


class FedCluster(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedCluster, self)._parse_kwargs(**kwargs)
        if ring := kwargs['ring']:
            self.group_num = ring.get('group_num', 5)
        self.pre_epoch = 20

    def _init_group_hook(self):
        self._init_state = deepcopy(self._model.state_dict())
        cids = FrozenList(self._fds)
        labels = self._cluster(cids)
        for gid in set(labels):
            self._groups[gid] = {
                'state': self._init_state,
                'clients': set([c for c, l in zip(cids, labels) if l == gid])
            }

    def _cluster(self, cids):
        M = self.kl_distance(cids, 1.)
        # _, labels, _ = cluster.k_means(M, self.group_num, random_state=self.seed)
        clf = KMeansConstrained(
            n_clusters=self.group_num,
            size_min=int(len(cids) / self.group_num * 0.75),
            size_max=int(len(cids) / self.group_num * 1.25),
            random_state=self.seed,
        )
        labels = clf.fit(M).labels_
        self._logger.info(f'cluster result: {labels}')
        return labels

    def _select_client(self):
        return reduce(concat, [
            random_select(
                FrozenList(self._groups[gid]['clients']),
                s_alpha=self.sample_rate,
                seed=self.seed + self._k
            )
            for gid in self._groups
        ])

    def kl_distance(self, cids, temp=1.):
        self._model.eval()
        dataloader = DataLoader(self._fds.secondary(10, 20), batch_size=10 * 20)
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.pre_epoch,
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._init_state, self._fds.train(c), args)
            for c in cids
        ])):
            self._model.load_state_dict(add(self._model.state_dict(), res[0]))
            for data, target in dataloader:
                self._cache[cid] = {
                    'logit': self._model(data) / temp
                }

        kl_dist = np.zeros((len(cids), len(cids)))
        for i, c1 in enumerate(cids):
            for j, c2 in enumerate(cids):
                kl_dist[i][j] = F.kl_div(self._cache[c1]['logit'], self._cache[c2]['logit']).detach().numpy()
        return kl_dist

    def _test(self):
        if self._k % self.test_step == 0:
            for gid in self._groups:
                cs = self._groups[gid]['clients']
                if len(cs) > 0:
                    self._metric_averager.reset()
                    self._val(cs)
                    self._handle_metric(self._metric_averager.compute(), 'test', self._writers[gid])
            self._logger.debug('\t'.join([
                f"{gid}: {sorted(self._groups[gid]['clients'])}"
                for gid in self._groups
            ]))
        self._model.load_state_dict(average([self._groups[gid]['state'] for gid in self._groups]))
        super(ClusteredFL, self)._test()




