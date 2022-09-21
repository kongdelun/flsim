import random
from datetime import datetime

import torch
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from trainer.algorithm.fedprox import ProxActor
from trainer.core.aggregator import StateAggregator
from trainer.core.proto import ClusteredFL, FedAvg
from trainer.util.metric import Metric, MetricAverager
from utils.cache import DiskCache
from utils.nn.functional import flatten, add
from utils.nn.init import with_kaiming_normal


class FedSem(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedSem, self)._parse_kwargs(**kwargs)
        if sem := kwargs['sem']:
            self.group_num = sem.get('group_num', 2)
            self.alpha = sem.get('alpha', 0.01)

    def _init_group(self):
        super(FedSem, self)._init_group()
        for i in range(self.group_num):
            self._groups[i] = {
                'clients': set(),
                'state': with_kaiming_normal(self._model.state_dict())
            }
        self._cache = DiskCache(
            self.cache_size,
            f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        )

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            ProxActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])
        self._metric_averager = MetricAverager()
        self._init_group()
        self._aggregators = {
            gid: StateAggregator()
            for gid in self._groups
        }
        self.writers = {
            gid: SummaryWriter(f'{self.writer.log_dir}/{gid}')
            for gid in self._groups
        }

    def _state(self, cid):
        gid = self._gid(cid)
        if gid is None:
            gid = random.choice(list(self._groups.keys()))
        # gid = np.argmin([len(self._groups[gid]['clients']) for gid in self._groups])
        return self._groups[gid]['state']

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), args)
            for c in cids
        ])):
            self._cache[cid] = {
                'state': add(self._state(cid), res[0]),
                'num_sample': res[1][0]
            }
            yield Metric(*res[1])

    def _best_group(self, cid):
        return torch.argmin(torch.tensor([
            torch.norm(flatten(self._cache[cid]['state']) - flatten(self._groups[gid]['state']), p=2).item()
            for gid in self._groups
        ])).item()

    def _aggregate(self, cids):
        for gid in self._groups:
            self._groups[gid]['clients'] -= set(cids)
        for cid in cids:
            gid = self._best_group(cid)
            self._groups[gid]['clients'].add(cid)
            self._aggregators[gid].update(**self._cache[cid])
        for gid in self._aggregators:
            try:
                self._groups[gid]['state'] = self._aggregators[gid].compute()
                self._aggregators[gid].reset()
            except AssertionError:
                continue
