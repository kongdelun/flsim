from abc import abstractmethod
from collections import OrderedDict
from importlib import import_module
from sys import modules

import torch
from trainer.algorithm.fedavg import FedAvg
from trainer.core.aggregator import BasicAggregator
from utils.nn.aggregate import average
from utils.nn.functional import zero_like


class FedOpt(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedOpt, self)._parse_kwargs(**kwargs)
        if adp := kwargs['adp']:
            self.global_opt = adp.get('global_opt', 'Adam')

    def _build_aggregator(self):
        return getattr(import_module(self.__module__), self.global_opt)(self._model.state_dict())

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()


class OptAggregator(BasicAggregator):

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

    def _compute_step(self):
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


class Yogi(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] -= (1 - self.beta_2) * self._d[ln] ** 2 * torch.sign(self._v[ln] - self._d[ln] ** 2)


class AdaGrad(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] += self._d[ln] ** 2


class Adam(OptAggregator):

    def _update_v(self):
        for ln in self._v:
            self._v[ln] = self.beta_2 * self._v[ln] + (1 - self.beta_2) * self._d[ln] ** 2


class test:

    def __init__(self):
        print(modules[__name__])
        print(test.__module__)
