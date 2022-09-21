from abc import ABC, abstractmethod
from typing import OrderedDict
from utils.nn.aggregate import fedavg


class NotCalculated(Exception):
    pass


class BaseAggregator(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class Aggregator(BaseAggregator):

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


class StateAggregator(Aggregator):

    def __init__(self):
        super(StateAggregator, self).__init__()
        self._states = []
        self._num_samples = []
        self._state = None

    def update(self, state: OrderedDict, num_sample):
        super(StateAggregator, self).update()
        self._states.append(state)
        self._num_samples.append(num_sample)

    def compute(self):
        try:
            return super(StateAggregator, self).compute()
        except NotCalculated:
            assert len(self._states) == len(self._num_samples) > 0
            self._res = fedavg(self._states, self._num_samples)
            return self._res

    def reset(self):
        super(StateAggregator, self).reset()
        self._num_samples.clear()
        self._states.clear()
