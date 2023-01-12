from typing import OrderedDict
from trainer.algorithm.fedavg import FedAvg
from trainer.core.aggregator import Aggregator
from utils.nn.aggregate import average
from utils.nn.functional import linear_sum, zero_like, add_


class FedAvgM(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvgM, self)._parse_kwargs(**kwargs)
        if avgm := kwargs['avgm']:
            self.beta = avgm.get('beta', 0.5)

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()

    def _build_aggregator(self):
        return AvgMAggregator(self._model.state_dict(), self.beta)


class AvgMAggregator(Aggregator):

    def __init__(self, state: OrderedDict, alpha: float):
        super(AvgMAggregator, self).__init__()
        self._state = state
        self._mom = zero_like(state)
        self._alpha = alpha
        self._grads = []
        self._num_samples = []

    def update(self, grad: OrderedDict, num_sample):
        super(AvgMAggregator, self).update()
        self._grads.append(grad)
        self._num_samples.append(num_sample)

    def reset(self):
        super(AvgMAggregator, self).reset()
        self._num_samples.clear()
        self._grads.clear()

    def _compute_step(self):
        assert len(self._grads) == len(self._num_samples) > 0
        grad = average(self._grads, self._num_samples)
        self._mom = linear_sum([self._mom, grad], [self._alpha, 1.])
        add_(self._state, self._mom)
        return self._state
