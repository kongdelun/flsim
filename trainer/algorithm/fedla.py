import dataclasses
from typing import OrderedDict
from trainer.algorithm.fedavg import FedAvg
from trainer.core.aggregator import Aggregator
from utils.nn.aggregate import average
from utils.nn.functional import state2vector, linear_sum, add, zero_like
from utils.nn.stats import l1_diversity


class FedLA(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if la := kwargs['la']:
            self.eps = la.get('eps', 0.03)
            self.delay_step = la.get('delay_step', 0)
            self.beta = la.get('beta', 0.5)

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()

    def _build_aggregator(self):
        return LAAggregator(
            self._model.state_dict(),
            max(1, int(self.sample_rate * len(self._fds))),
            self.beta,
            self.eps,
            self.delay_step
        )

    def _local_update_args(self, cids):
        return [(s, self._fds.train(c), self.local_args) for c, s in zip(cids, self._aggregator.states())]


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

    def reset(self):
        super(LAAggregator, self).reset()
        self._num_samples.clear()
        self._grads.clear()

    def _compute_step(self):
        assert len(self._grads) == self._num_parallel > 0
        for s, n, g in zip(self._s, self._num_samples, self._grads):
            s.num += n
            s.momentum = linear_sum([s.momentum, g], [self._beta, 1.])
            s.state = add(s.state, s.momentum)
        self._k += 1
        # 判断是否聚合
        if self.__check():
            self._state = average([s.state for s in self._s], [s.num for s in self._s])
            self._s = [self.Status(s.momentum, self._state) for s in self._s]
        return self._state

    def __check(self):
        if self._delay_step > 0:
            if self._k % self._delay_step == 0:
                return True
        else:
            # 计算差异度
            vecs = state2vector([s.state for s in self._s])
            prev_diff = self._diff
            self._diff = l1_diversity(vecs).item()
            diff_ratio = abs(self._diff - prev_diff) / self._diff
            # 判断是否需要聚合
            if diff_ratio < self._eps:
                return True
        return False

    def states(self):
        return [s.state for s in self._s]
