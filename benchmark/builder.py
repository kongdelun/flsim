import importlib
import traceback
from typing import Optional
from torch.nn import Module
from utils.data.dataset import FederatedDataset


def get_trainer(name: str, net: Module, fds: FederatedDataset, cfg: dict):
    mod = None
    for md in [
        f'trainer.core.proto',
        f'trainer.algorithm.{name.lower()}',
        f'trainer.algorithm.cfl.{name.lower()}'
    ]:
        try:
            mod = importlib.import_module(md)
        except ModuleNotFoundError:
            continue
    try:
        cls = getattr(mod, name)
        ret = cls(net, fds, **cfg)
    except:
        print(traceback.format_exc())
        raise ImportError(f'No such trainer: {name}')
    return ret


def build_model(name, args: Optional[dict] = None):
    if args is None:
        args = dict()
    for md in [
        f'benchmark.model.cv',
        f'benchmark.model.nlp',
    ]:
        try:
            return getattr(importlib.import_module(md), name)(**args)
        except RuntimeError:
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
        except RuntimeError:
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
        except:
            continue
    else:
        raise ImportError(f'No such trainer: {name}')
