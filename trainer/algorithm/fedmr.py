from typing import OrderedDict
from trainer.core.aggregator import Aggregator
from utils.nn.aggregate import shuffle_layer, average


class ShuffleAggregator(Aggregator):

    def __init__(self, state: OrderedDict, delay_step: int = 5):
        super(ShuffleAggregator, self).__init__()
        self._state = state
        self._delay_step = delay_step
        self._k = -1
        self._num_samples = []
        self._states = []

    def update(self, state: OrderedDict, num_sample):
        super(ShuffleAggregator, self).update()
        self._states.append(state)
        self._num_samples.append(num_sample)

    def reset(self):
        super(ShuffleAggregator, self).reset()
        self._num_samples.clear()
        self._states.clear()

    def _has_agg(self):
        if self._delay_step > 0:
            if self._k % self._delay_step == 0:
                return True
        return False

    def _adapt_fn(self):
        self._k += 1
        if not self._has_agg():
            shuffle_layer(self._states)
        else:
            self._state = average(self._states, self._num_samples)
        return self._state
