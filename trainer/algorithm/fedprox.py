from collections import OrderedDict

import ray
import torch
from ray.util import ActorPool
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from trainer.core.actor import CPUActor
from trainer.core.proto import FedAvg
from utils.nn.functional import flatten, sub


@ray.remote
class ProxActor(CPUActor):

    def _setup(self, args: dict):
        self._batch_size = args.get('batch_size', 32)
        self._epoch = args.get('epoch', 5)
        self._max_grad_norm = args.get('max_grad_norm', 10.0)
        self._opt = args.get('opt', {'lr': 0.001})

    def __init__(self, model: Module, loss: Module, alpha: float):
        super().__init__(model, loss)
        self._alpha = alpha

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self._setup(args)
        self.set_state(state)
        opt = self.opt_fn(**self._opt)
        self.model.train()
        for k in range(self._epoch):
            for data, target in self.dataloader(dataset, self._batch_size):
                opt.zero_grad()
                loss = self.loss(self.model(data), target) + self.__rt(state)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, self._batch_size)

    def __rt(self, global_state: OrderedDict):
        return .5 * self._alpha * torch.sum(torch.pow(flatten(self.get_state()) - flatten(global_state), 2))


class FedProx(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if prox := kwargs['prox']:
            self.alpha = prox.get('alpha', 0.01)

    def _configure_actor_pool(self):
        self._pool = ActorPool([
            ProxActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])

