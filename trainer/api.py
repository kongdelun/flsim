import importlib
import traceback

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
        trainer = cls(net, fds, **cfg)
    except:
        print(traceback.format_exc())
        raise ImportError(f'No such trainer: {name}')
    return trainer
