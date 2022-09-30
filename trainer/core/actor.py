from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy

import ray
import torch
from torch import optim
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchmetrics import SumMetric, MeanMetric, Accuracy

from utils.nn.functional import sub
from utils.tool import os_platform


class CPUActor:

    def __init__(self, model: Module, loss: Module, local_opt: str = "sgd"):
        self.model = model
        self.loss = loss
        self.local_opt = local_opt
        self._num_workers, self._prefetch_factor = 0, 2
        if 'linux' in os_platform():
            self._num_workers, self._prefetch_factor = 2, 4

    def opt_fn(self, **args):
        if self.local_opt == 'sgd':
            return optim.SGD(self.model.parameters(), **args)
        elif self.local_opt == 'adam':
            return optim.Adam(self.model.parameters(), **args)
        return None

    def dataloader(self, dataset: Dataset, batch_size: int):
        for data, target in DataLoader(
                dataset, batch_size,
                prefetch_factor=self._prefetch_factor,
                num_workers=self._num_workers
        ):
            yield data, target

    def get_state(self, copy=False):
        if not copy:
            return self.model.state_dict()
        return deepcopy(self.model.state_dict())

    def set_state(self, state: OrderedDict):
        self.model.load_state_dict(state)

    @abstractmethod
    def _setup(self, args: dict):
        raise NotImplementedError

    @abstractmethod
    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, state: OrderedDict, dataset: Dataset, batch_size: int):
        num, loss, acc = SumMetric(), MeanMetric(), Accuracy()
        self.model.load_state_dict(state)
        self.model.eval()
        for data, target in self.dataloader(dataset, batch_size):
            logit = self.model(data)
            num.update(target.shape[0])
            loss.update(self.loss(logit, target))
            acc.update(logit, target)
        return num.compute().item(), loss.compute().item(), acc.compute().item()


@ray.remote
class SGDActor(CPUActor):

    def _setup(self, args: dict):
        self._batch_size = args.get('batch_size', 32)
        self._epoch = args.get('epoch', 5)
        self._max_grad_norm = args.get('max_grad_norm', 10.0)
        self._opt = args.get('opt', {'lr': 0.001})

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self._setup(args)
        self.set_state(state)
        opt = self.opt_fn(**self._opt)
        self.model.train()
        for k in range(self._epoch):
            for data, target in self.dataloader(dataset, self._batch_size):
                opt.zero_grad()
                self.loss(self.model(data), target).backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, self._batch_size)
