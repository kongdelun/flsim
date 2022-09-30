import importlib
import logging
from typing import Optional, Iterator

from torch.nn import Module, Parameter

from utils.data.dataset import FederatedDataset

logger = logging.getLogger(__name__)


def build_model(name, args: Optional[dict] = None):
    if args is None:
        args = dict()
    for md in [
        f'benchmark.model.cv',
        f'benchmark.model.nlp',
    ]:
        try:
            return getattr(importlib.import_module(md), name)(**args)
        except AttributeError as e:
            logger.debug(e)
            continue
    else:
        raise ImportError(f'No such model: {name}')


def build_federated_dataset(name: str, args: dict):
    if args is None:
        args = dict()
    for md in [
        f'benchmark.ctx'
    ]:
        try:
            return getattr(importlib.import_module(md), name)(**args)
        except AttributeError as e:
            logger.debug(e)
            continue
    else:
        raise ImportError(f'No such dataset: {name}')


def build_trainer(name: str, net: Module, fds: FederatedDataset, args: dict):
    if args is None:
        args = dict()
    for md in [
        f'trainer.core.proto',
        f'trainer.algorithm.{name.lower()}',
        f'trainer.algorithm.cfl.{name.lower()}'
    ]:
        try:
            return getattr(importlib.import_module(md), name)(net, fds, **args)
        except AttributeError as e:
            logger.debug(e)
            continue
    else:
        raise ImportError(f'No such trainer: {name}')


def build_optimizer(name: str, param: Iterator[Parameter], args: dict):
    if args is None:
        args = dict(lr=0.01)
    for md in [
        f'torch.optim',
        f'util.optim',
    ]:
        try:
            return getattr(importlib.import_module(md), name)(param, **args)
        except AttributeError as e:
            logger.debug(e)
            continue
    else:
        raise ImportError(f'No such Optimizer: {name}')
