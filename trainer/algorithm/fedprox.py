from collections import OrderedDict

import ray
import torch
from ray.util import ActorPool
from torch import optim

from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import Dataset
from trainer.core.actor import CPUActor
from trainer.core.aggregator import StateAggregator
from trainer.core.proto import FedAvg
from trainer.util.metric import MetricAverager
from utils.nn.functional import flatten, sub


@ray.remote
class ProxActor(CPUActor):

    def __init__(self, model: Module, loss: Module, alpha: float):
        super().__init__(model, loss)
        self._alpha = alpha

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        opt = args.get('opt', {'lr': 0.001})
        batch_size = args.get('batch_size', 32)
        epoch = args.get('epoch', 5)
        self.set_state(state)
        opt = optim.SGD(self.model.parameters(), **opt)
        self.model.train()
        for k in range(epoch):
            for data, target in self.dataloader(dataset, batch_size):
                opt.zero_grad()
                loss = self.loss(self.model(data), target) + self.__fix_term(state)
                loss.backward()
                opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, batch_size)

    def __fix_term(self, global_state: OrderedDict):
        return .5 * self._alpha * torch.sum(torch.pow(flatten(self.get_state()) - flatten(global_state), 2))


class FedProx(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if prox := kwargs['prox']:
            self.alpha = prox.get('alpha', 0.01)

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            ProxActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(self.actor_num)
        ])
        self._aggregator = StateAggregator()
        self._metric_averager = MetricAverager()

