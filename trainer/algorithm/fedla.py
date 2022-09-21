import dataclasses
from typing import OrderedDict

import frozenlist
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss

from trainer.core.actor import SGDActor
from trainer.core.aggregator import Aggregator, NotCalculated
from trainer.core.proto import FedAvg
from trainer.util.metric import Metric, MetricAverager
from utils.nn.aggregate import fedavg
from utils.nn.functional import state2vector, linear_sum, add, zero_like
from utils.nn.stats import diff


class LAAggregator(Aggregator):
    @dataclasses.dataclass
    class Status:
        momentum: OrderedDict
        state: OrderedDict
        num: int = 0

    def __init__(self, state: OrderedDict, num_parallel: int, beta: float, eps: float, delay_step: int = 0):
        super(LAAggregator, self).__init__()
        self._num_samples = []
        self._grads = []
        self._state = state
        self._num_parallel = num_parallel
        self._eps = eps
        self._beta = beta
        self._delay_step = delay_step
        self._k = -1
        self._diff = 0.
        self._s = [
            self.Status(zero_like(state), state)
            for _ in range(num_parallel)
        ]

    def update(self, grad: OrderedDict, num_sample):
        super(LAAggregator, self).update()
        self._grads.append(grad)
        self._num_samples.append(num_sample)

    def _has_agg(self):
        if self._delay_step > 0:
            if self._k % self._delay_step == 0:
                return True
        else:
            # 计算差异度
            vecs = state2vector([s.state for s in self._s])
            prev_diff = self._diff
            self._diff = diff(vecs).item()
            diff_ratio = abs(self._diff - prev_diff) / self._diff
            # 判断是否需要聚合
            if diff_ratio < self._eps:
                return True
        return False

    def compute(self):
        try:
            return super(LAAggregator, self).compute()
        except NotCalculated:
            assert len(self._grads) == self._num_parallel > 0
            for s, n, g in zip(self._s, self._num_samples, self._grads):
                s.num += n
                s.momentum = linear_sum([s.momentum, g], [self._beta, 1.])
                s.state = add(s.state, s.momentum)
            self._k += 1
            # 判断是否聚合
            if self._has_agg():
                self._state = fedavg([s.state for s in self._s], [s.num for s in self._s])
                self._s = [self.Status(s.momentum, self._state) for s in self._s]
            self._res = self._state
            return self._res

    def reset(self):
        super(LAAggregator, self).reset()
        self._num_samples.clear()
        self._grads.clear()

    @property
    def states(self):
        return frozenlist.FrozenList(s.state for s in self._s)


class FedLA(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if la := kwargs['la']:
            self.eps = la.get('eps', 0.03)
            self.delay_step = la.get('delay_step', 0)
            self.beta = la.get('beta', 0.5)

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            SGDActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])
        self._aggregator = LAAggregator(
            self._model.state_dict(),
            int(max(1, self.sample_rate * len(self._fds))),
            self.beta,
            self.eps,
            self.delay_step
        )
        self._metric_averager = MetricAverager()

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch
        }
        for res in self._pool.map(lambda a, v: a.fit.remote(*v), [
            (s, self._fds.train(cid), args)
            for cid, s in zip(cids, self._aggregator.states)
        ]):
            self._aggregator.update(res[0], res[1][0])
            yield Metric(*res[1])

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
