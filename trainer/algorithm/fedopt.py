from abc import abstractmethod
from collections import OrderedDict

import torch
from trainer.core.aggregator import StateAggregator
from trainer.core.proto import FedAvg
from utils.nn.aggregate import average
from utils.nn.functional import zero_like, sub


class OptAggregator(StateAggregator):

    def __init__(self, state: OrderedDict, beta_1=0.9, beta_2=0.99, eta=1e-2, tau=1e-3):
        super(OptAggregator, self).__init__()
        self._state = state
        self._m = zero_like(state)
        self._d = zero_like(state)
        self._v = zero_like(state)
        self._agg = state
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eta = eta
        self.tau = tau

    def _adapt_fn(self):
        assert len(self._states) == len(self._num_samples) > 0
        self._d = average(self._states)
        for ln in self._m:
            self._m[ln] = self.beta_1 * self._m[ln] + (1 - self.beta_1) * self._d[ln]
        self._update_v()
        for ln in self._state:
            self._state[ln] += self.eta * self._m[ln] / (torch.sqrt(self._v[ln]) + self.tau)
        return self._state

    @abstractmethod
    def _update_v(self):
        raise NotImplementedError


class YogiAggregator(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] -= (1 - self.beta_2) * self._d[ln] ** 2 * torch.sign(self._v[ln] - self._d[ln] ** 2)


class AdaGradAggregator(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] += self._d[ln] ** 2


class AdamAggregator(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] = self.beta_2 * self._v[ln] + (1 - self.beta_2) * self._d[ln] ** 2


class FedOpt(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedOpt, self)._parse_kwargs(**kwargs)
        if opt := kwargs['fedopt']:
            self.global_opt = opt.get('global_opt', 'Adam')

    def _configure_aggregator(self):
        if self.global_opt == 'Adam':
            self._aggregator = AdamAggregator(self._model.state_dict())
        elif self.global_opt == 'Yogi':
            self._aggregator = YogiAggregator(self._model.state_dict())
        elif self.global_opt == 'AdaGrad':
            self._aggregator = AdaGradAggregator(self._model.state_dict())
        else:
            raise ValueError(f'Unknown global optimizer: {self.global_opt}')

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
