import importlib

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
    if mod is None:
        raise ModuleNotFoundError
    cls = getattr(mod, name)
    return cls(net, fds, **cfg)
