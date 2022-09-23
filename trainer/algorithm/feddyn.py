from datetime import datetime
from typing import OrderedDict

import ray
import torch
from ray.util import ActorPool
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import Dataset

from trainer.core.actor import CPUActor
from trainer.core.aggregator import Aggregator, NotCalculated
from trainer.core.proto import FedAvg
from utils.cache import DiskCache
from utils.metric import Metric
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

    def compute(self):
        try:
            return super(DynAggregator, self).compute()
        except NotCalculated:
            assert len(self._states) > 0
            m, n = self._num_clients, len(self._states)
            sum_theta = linear_sum(self._states)
            for ln in self._state:
                self._h[ln] -= self._alpha / m * (sum_theta[ln] - n * self._state[ln])
                self._state[ln] = 1. / n * sum_theta[ln] - 1. / self._alpha * self._h[ln]
            self._res = self._state
            return self._res

    def reset(self):
        super(DynAggregator, self).reset()
        self._states.clear()


@ray.remote
class DynActor(CPUActor):

    def __init__(self, model: Module, loss: Module, alpha: float):
        super().__init__(model, loss)
        self._alpha = alpha

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        opt = args.get('opt', {'lr': 0.001})
        batch_size = args.get('batch_size', 32)
        epoch = args.get('epoch', 5)
        grad = args.get('grad', None)
        if grad is None:
            grad = zero_like(state)
        self.set_state(state)
        opt = optim.SGD(self.model.parameters(), **opt)
        self.model.train()
        for k in range(epoch):
            for data, target in self.dataloader(dataset, batch_size):
                opt.zero_grad()
                loss = self.loss(self.model(data), target) + self.__rt(state, grad)
                loss.backward()
                opt.step()
        state_ = self.get_state(copy=True)
        for ln in grad:
            grad[ln] -= self._alpha * (state_[ln] - state[ln])
        return state_, grad, self.evaluate(state_, dataset, batch_size)

    def __rt(self, global_state: OrderedDict, grad: OrderedDict):
        state = self.get_state()
        l1 = torch.dot(flatten(grad), flatten(state))
        l2 = .5 * self._alpha * torch.sum(torch.pow(flatten(state) - flatten(global_state), 2))
        return -l1 + l2


class FedDyn(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if dyn := kwargs['dyn']:
            self.alpha = dyn.get('alpha', 0.01)

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            DynActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])
        self._aggregator = DynAggregator(
            self._model.state_dict(),
            len(self._fds), self.alpha
        )
        self._cache = DiskCache(
            self.cache_size,
            f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        )

    def _local_update_callback(self, cid, res):
        self._cache[cid] = res[1]
        self._aggregator.update(res[0])

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch
        }
        for res, cid in zip(self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), dict({'grad': self._cache.get(c)}, **args))
            for c in cids
        ]), cids):
            self._local_update_callback(cid, res)
            self._metric_averager.update(Metric(*res[2]))

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
