from typing import OrderedDict

import ray
import torch
from ray.util import ActorPool
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from trainer.core.actor import CPUActor
from trainer.core.aggregator import Aggregator
from trainer.core.proto import FedAvg
from utils.nn.functional import flatten, zero_like, linear_sum


class DynAggregator(Aggregator):

    def __init__(self, state: OrderedDict, num_clients: int, alpha: float):
        super(DynAggregator, self).__init__()
        self._states = []
        self._alpha = alpha
        self._num_clients = num_clients
        self._state = state
        self._h = zero_like(state)

    def update(self, state: OrderedDict):
        super(DynAggregator, self).update()
        self._states.append(state)

    def reset(self):
        super(DynAggregator, self).reset()
        self._states.clear()

    def _adapt_fn(self, m, n):
        assert len(self._states) > 0
        m, n = self._num_clients, len(self._states)
        sum_theta = linear_sum(self._states)
        for ln in self._state:
            self._h[ln] -= self._alpha / m * (sum_theta[ln] - n * self._state[ln])
            self._state[ln] = 1. / n * sum_theta[ln] - 1. / self._alpha * self._h[ln]
        return self._state


@ray.remote
class DynActor(CPUActor):

    def __init__(self, model: Module, loss: Module, alpha: float):
        super(DynActor, self).__init__(model, loss)
        self._alpha = alpha

    def _setup(self, args: dict):
        self._batch_size = args.get('batch_size', 32)
        self._epoch = args.get('epoch', 5)
        self._max_grad_norm = args.get('max_grad_norm', 10.0)
        self._opt = args.get('opt', {'lr': 0.001})
        self._grad = args.get('grad', None)
        if self._grad is None:
            self._grad = zero_like(self.get_state())

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self._setup(args)
        self.set_state(state)
        opt = self.opt_fn(self._opt)
        self.model.train()
        for k in range(self._epoch):
            for data, target in self.dataloader(dataset, self._batch_size):
                opt.zero_grad()
                loss = self.loss(self.model(data), target) + self.__rt(state)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                opt.step()
        state_ = self.get_state(copy=True)
        for ln in self._grad:
            self._grad[ln] -= self._alpha * (state_[ln] - state[ln])
        return state_, self.evaluate(state_, dataset, self._batch_size), self._grad

    def __rt(self, global_state: OrderedDict):
        state = self.get_state()
        l1 = torch.dot(flatten(self._grad), flatten(state))
        l2 = .5 * self._alpha * torch.sum(torch.pow(flatten(state) - flatten(global_state), 2))
        return -l1 + l2


class FedDyn(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if dyn := kwargs['dyn']:
            self.alpha = dyn.get('alpha', 0.01)

    def _configure_actor_pool(self):
        self._pool = ActorPool([
            DynActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])

    def _configure_aggregator(self):
        self._aggregator = DynAggregator(
            self._model.state_dict(),
            len(self._fds), self.alpha
        )

    def _local_update_hook(self, cid, res):
        self._cache[cid] = res[2]
        self._aggregator.update(res[0])

    def _local_update_setup(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'max_grad_norm': self.max_grad_norm
        }
        return [(self._state(c), self._fds.train(c), dict({'grad': self._cache.get(c)}, **args)) for c in cids]

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
