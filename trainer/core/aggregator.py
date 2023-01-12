from abc import abstractmethod
from typing import OrderedDict
from utils.nn.aggregate import average
from utils.nn.functional import linear_sum, scalar_mul_


class Aggregator:

    def __init__(self):
        self._res = None

    def update(self, *args, **kwargs):
        if self._res:
            raise RuntimeError(f'Please reset {self.__class__.__name__} !')

    def compute(self):
        if self._res is None:
            self._res = self._compute_step()
        return self._res

    def reset(self):
        self._res = None

    @abstractmethod
    def _compute_step(self) -> OrderedDict:
        raise NotImplementedError


class BasicAggregator(Aggregator):

    def __init__(self):
        super(BasicAggregator, self).__init__()
        self._states = []
        self._num_samples = []

    def update(self, state: OrderedDict, num_sample: int):
        super(BasicAggregator, self).update()
        self._states.append(state)
        self._num_samples.append(num_sample)

    def reset(self):
        super(BasicAggregator, self).reset()
        self._num_samples.clear()
        self._states.clear()

    def _compute_step(self):
        assert len(self._states) == len(self._num_samples) > 0
        return average(self._states, self._num_samples)


class OnlineAggregator(Aggregator):

    def __init__(self):
        super(OnlineAggregator, self).__init__()
        self._m, self._c = None, None

    def update(self, state: OrderedDict, weight: float):
        super(OnlineAggregator, self).update()
        if self._m is None:
            self._m = state
            self._c = weight
        else:
            self._m = scalar_mul_(
                linear_sum([self._m, state], [self._c, weight]),
                1 / (self._c + weight)
            )
            self._c += weight

    def _compute_step(self) -> OrderedDict:
        return self._m

    def reset(self, over=False):
        super(OnlineAggregator, self).reset()
        if over is True:
            self._m, self._c = None, None
