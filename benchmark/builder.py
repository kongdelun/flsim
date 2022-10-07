import importlib
from traceback import format_exc
from typing import Optional, Iterator
from torch.nn import Module, Parameter
from utils.data.dataset import FederatedDataset
from utils.logger import Logger

logger = Logger.get_logger(__name__)


def build_model(name, args: Optional[dict] = None):
    if args is None:
        args = dict()
    for md in [
        f'benchmark.model.cv',
        f'benchmark.model.nlp',
    ]:
        try:
            model = getattr(importlib.import_module(md), name)(**args)
            logger.info(f'{md}.{name} has created !')
            return model
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue
        except:
            logger.warning(format_exc())
            raise ImportError(f'No such model: {name}')


def build_federated_dataset(name: str, args: dict):
    if args is None:
        args = dict()
    for md in [
        f'benchmark.ctx'
    ]:
        try:
            federated_dataset = getattr(importlib.import_module(md), name)(**args)
            logger.info(f'{md}.{name} has created !')
            return federated_dataset
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue
        except:
            logger.warning(format_exc())
            raise ImportError(f'No such federated dataset: {name}')


def build_trainer(name: str, net: Module, fds: FederatedDataset, args: dict):
    if args is None:
        args = dict()
    for md in [
        f'trainer.core.proto',
        f'trainer.algorithm.{name.lower()}',
        f'trainer.algorithm.cfl.{name.lower()}'
    ]:
        try:
            trainer = getattr(importlib.import_module(md), name)(net, fds, **args)
            logger.info(f'{md}.{name} has created !')
            return trainer
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue
        except:
            logger.warning(format_exc())
            raise ImportError(f'No such trainer: {name}')


def build_optimizer(name: str, param: Iterator[Parameter], args: dict):
    if args is None:
        args = dict(lr=0.01)
    for md in [
        f'torch.optim',
        f'util.optim',
    ]:
        try:
            optimizer = getattr(importlib.import_module(md), name)(param, **args)
            logger.info(f'{md}.{name} has created !')
            return optimizer
        except ModuleNotFoundError:
            continue
        except AttributeError:
            continue
        except:
            logger.warning(format_exc())
            raise ImportError(f'No such optimizer: {name}')
