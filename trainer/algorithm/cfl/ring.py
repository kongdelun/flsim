import random
from copy import deepcopy
from functools import reduce
from operator import concat

import numpy as np
import torch
import torch.nn.functional as F
from frozenlist import FrozenList
from k_means_constrained import KMeansConstrained
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from trainer.algorithm.fedprox import ProxActor
from trainer.core.proto import ClusteredFL
from utils.compressor.basic import TopkCompressor
from utils.metric import average
from utils.nn.functional import zero_like, add_, add, flatten, linear_sum
from utils.nn.stats import cosine
from utils.select import random_select


class Ring(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(Ring, self)._parse_kwargs(**kwargs)
        if ring := kwargs['ring']:
            self.rho = ring.get('rho', 0.7)
            self.alpha = ring.get('alpha', 0.0005)
            self.compress_ratio = ring.get('compress_ratio', 0.6)
            self.group_num = ring.get('group_num', 8)
        self.pre_epoch = 20

    def _configure_actor_pool(self):
        self._pool = ActorPool([
            ProxActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])

    def _init_group_hook(self):
        self._cur = -1
        self._compressor = TopkCompressor(self.compress_ratio)
        self._mom = zero_like(self._model.state_dict())
        cids = FrozenList(self._fds)
        labels = self._cluster(cids)
        for gid in set(labels):
            self._groups[gid] = {
                'state': deepcopy(self._model.state_dict()),
                'clients': set([c for c, l in zip(cids, labels) if l == gid]),
                'loss': 0.0
            }

    def _cluster(self, cids):
        # return [i % self.group_num for i in range(len(cids))]
        M = self.kl_distance(cids, 1.)
        clf = KMeansConstrained(
            n_clusters=self.group_num,
            size_min=int(len(cids) / self.group_num * 0.75),
            size_max=int(len(cids) / self.group_num * 1.25),
            random_state=self.seed,
        )
        labels = clf.fit(M).labels_
        # _, labels, _ = cluster.k_means(M, self.group_num, random_state=self.seed)
        self._logger.info(f'cluster result: {labels}')
        return labels

    def _select_client(self):
        # self._cur = (self._cur + 1) % len(self._groups)
        max_loss = -1
        for gid in self._groups:
            loss = self._groups[gid]['loss']
            if max_loss < loss:
                max_loss = loss
                self._cur = gid
        return random_select(
            FrozenList(self._groups[self._cur]['clients']),
            s_alpha=self.sample_rate,
            seed=self.seed + self._k
        )

    def _compress(self, state):
        values, indices = self._compressor.compress(state)
        return self._compressor.decompress(values, indices, state)

    def _aggregate(self, cids):
        grad = self._aggregators[self._cur].compute()
        self._mom = linear_sum([self._mom, grad], [self.rho, 1.])
        add_(self._groups[self._cur]['state'], self._mom)
        for gid in self._groups:
            self._groups[gid]['state'] = self._groups[self._cur]['state']
        self._aggregators[self._cur].reset()

    def _local_update_hook(self, cid, res):
        self._compress(res[0])
        self._cache[cid] = res[0]
        self._aggregators[self._cur].update(res[0], res[1][0])

    def kl_distance(self, cids, temp=1.):

        def pretrain():
            args = {
                'opt': self.opt,
                'batch_size': self.batch_size,
                'epoch': self.epoch
            }
            for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
                (self._model.state_dict(), self._fds.train(c), args)
                for c in cids
            ])):
                self._cache[cid] = dict(state=add(self._model.state_dict(), self._compress(res[0])))

        @torch.no_grad()
        def logit():
            self._model.eval()
            for cid in cids:
                self._model.load_state_dict(self._cache[cid]['state'])
                for data, target in DataLoader(self._fds.secondary(10, 200), batch_size=10 * 20):
                    self._cache[cid]['logit'] = self._model(data) / temp

        pretrain()
        logit()
        kl_dist = np.zeros((len(cids), len(cids)))
        for i, c1 in enumerate(cids):
            for j, c2 in enumerate(cids):
                kl_dist[i][j] = F.kl_div(self._cache[c1]['logit'], self._cache[c2]['logit']).numpy()
        return kl_dist

    def _test(self):
        if self._k % self.test_step == 0:
            metrics = []
            for gid in self._groups:
                cs = self._groups[gid]['clients']
                if len(cs) > 0:
                    self._metric_averager.reset()
                    self._val(cs)
                    self._handle_metric(self._metric_averager.compute(), 'test', self._writers[gid])
                    m = self._metric_averager.compute()
                    metrics.append(m)
                    self._groups[gid]['loss'] = m.loss

            self._handle_metric(average(metrics), 'test')
            self._logger.debug('\t'.join([
                f"{gid}: {sorted(self._groups[gid]['clients'])}"
                for gid in self._groups
            ]))