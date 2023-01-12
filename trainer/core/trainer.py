import gc
from abc import abstractmethod
from traceback import format_exc

import ray
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from trainer.utils.metric import Metric
from utils.data.dataset import FederatedDataset
from utils.logger import Logger
from utils.result import print_banner
from utils.tool import set_seed, locate


class FLTrainer:

    def __init__(self, model: Module, fds: FederatedDataset, **kwargs):
        self._fds = fds
        self._model = model
        self._parse_kwargs(**kwargs)
        print_banner(self.__class__.__name__)

    def _parse_kwargs(self, **kwargs):
        self.name = f"{self.__class__.__name__}{kwargs.get('tag', '')}"
        self.actor_num = kwargs.get('actor_num', 5)
        self.seed = kwargs.get('seed', 2077)
        self.sample_rate = kwargs.get('sample_rate', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epoch = kwargs.get('epoch', 5)
        self.test_step = kwargs.get('test_step', 5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 10.0)
        self.opt = kwargs.get('opt', {'lr': 0.002})
        self.round = kwargs.get('round', 300)
        self.cache_size = kwargs.get('cache_size', 3)
        self.local_args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'max_grad_norm': self.max_grad_norm
        }

    # 初始化
    def _init(self):
        set_seed(self.seed)
        ray.init(configure_logging=False)
        self._logger = Logger.get_logger(self.name)
        self._k = 0

    def _handle_metric(self, metric: Metric, tag: str, writer: SummaryWriter = None):
        suffix = writer.log_dir.split('/')[-1]
        self._logger.info(f"[{self._k}] {tag.capitalize()}{'' if suffix == self.name else f'({suffix})'}: {metric}")
        if writer is not None:
            if suffix != self.name and tag in ['test']:
                tag = 'group'
            writer.add_scalar(f'{tag}/acc', metric.acc, self._k)
            writer.add_scalar(f'{tag}/loss', metric.loss, self._k)
            writer.flush()

    def _step(self):
        self._k += 1

    def _clean(self):
        ray.shutdown()
        gc.collect()

    @abstractmethod
    def _state(self, cid):
        raise NotImplementedError

    @abstractmethod
    def _select_client(self):
        raise NotImplementedError

    @abstractmethod
    def _local_update(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _val(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _test(self):
        raise NotImplementedError

    def _train(self):
        # 1.选择参与设备
        selected = self._select_client()
        self._logger.info(f'[{self._k}] Selected: {selected}')
        # 2.本地训练
        self._local_update(selected)
        # 3.聚合更新
        self._aggregate(selected)
        # 4.聚合模型验证
        self._val(selected)
        # 5.模型测试
        if self._k % self.test_step == 0:
            self._test()

    def start(self):
        self._init()
        try:
            while self._k <= self.round:
                self._train()
                self._step()
        except:
            self._logger.warning(format_exc())
        finally:
            self._clean()


def build_trainer(name: str, net: Module, fds: FederatedDataset, args: dict):
    args = dict(model=net, fds=fds, **args)
    return locate(
        [
            f'trainer.algorithm.{name.lower()}',
            f'trainer.algorithm.cluster.{name.lower()}'
        ],
        name, args
    )
