from abc import abstractmethod
from typing import OrderedDict
from utils.nn.aggregate import average


class NotCalculated(Exception):
    pass


class BaseAggregator:

    def __init__(self):
        self._res = None

    def update(self, *args, **kwargs):
        if self._res:
            raise RuntimeError(f'Please reset {self.__class__.__name__} !')

    def compute(self):
        if self._res is None:
            raise NotCalculated(self._res)
        return self._res

    def reset(self):
        self._res = None


class Aggregator(BaseAggregator):

    def compute(self):
        try:
            return super(Aggregator, self).compute()
        except NotCalculated:
            self._res = self._adapt_fn()
            return self._res

    @abstractmethod
    def _adapt_fn(self):
        raise NotImplementedError


class StateAggregator(Aggregator):

    def __init__(self):
        super(StateAggregator, self).__init__()
        self._states = []
        self._num_samples = []

    def update(self, state: OrderedDict, num_sample):
        super(StateAggregator, self).update()
        self._states.append(state)
        self._num_samples.append(num_sample)

    def reset(self):
        super(StateAggregator, self).reset()
        self._num_samples.clear()
        self._states.clear()

    def _adapt_fn(self):
        assert len(self._states) == len(self._num_samples) > 0
        return average(self._states, self._num_samples)
