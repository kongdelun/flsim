import gc
import traceback
from abc import abstractmethod

import ray
from ray.util import ActorPool
from torch.nn import Module, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from env import TB_OUTPUT
from trainer.core.actor import SGDActor
from trainer.core.aggregator import StateAggregator
from utils.data.dataset import FederatedDataset
from utils.metric import Metric, MetricAverager, average
from utils.nn.functional import add, add_
from utils.nn.init import with_kaiming_normal
from utils.result import progress_bar, print_banner
from utils.select import random_select
from utils.tool import set_seed


class FLTrainer:

    def __init__(self, model: Module, fds: FederatedDataset, **kwargs):
        self._fds = fds
        self._model = model
        self._parse_kwargs(**kwargs)
        if self.verbose:
            print_banner(self.__class__.__name__)
            summary(self._model)

    def _parse_kwargs(self, **kwargs):
        self.name = f"{self.__class__.__name__}{kwargs.get('tag', '')}"
        self.verbose = kwargs.get('verbose', True)
        self.actor_num = kwargs.get('actor_num', 5)
        self.seed = kwargs.get('seed', 2077)
        self.sample_rate = kwargs.get('sample_rate', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epoch = kwargs.get('epoch', 5)
        self.test_step = kwargs.get('test_step', 5)
        self.opt = kwargs.get('opt', {'lr': 0.002})
        self.round = kwargs.get('round', 300)
        self.cache_size = kwargs.get('cache_size', 3)
        self.writer = kwargs.get('writer', SummaryWriter(TB_OUTPUT + self.name))

    def _init(self):
        self._k = 0
        set_seed(self.seed)
        self._bar = progress_bar(self.round, 'Training:')
        self._model.load_state_dict(
            with_kaiming_normal(self._model.state_dict())
        )
        self._metric_averager = MetricAverager()

    def _print_msg(self, msg):
        if not isinstance(msg, str):
            msg = str(msg)
        if self.verbose:
            if self._bar:
                self._bar.write(msg)
            else:
                print(msg)

    def _write_tb(self, tag, metric: Metric, writer: SummaryWriter = None):
        if writer is None:
            writer = self.writer
        writer.add_scalar(f'{tag}/acc', metric.acc, self._k)
        writer.add_scalar(f'{tag}/loss', metric.loss, self._k)
        writer.flush()

    def _update_iter(self):
        self._k += 1
        self._print_msg('=' * 65)
        self._print_msg(f'Round: {self._k}')
        if self._bar:
            self._bar.update()

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

    @abstractmethod
    def start(self):
        raise NotImplementedError

    def close(self):
        ray.shutdown()
        gc.collect()


class FedAvg(FLTrainer):

    def _log_metric(self, metric: Metric, tag: str, writer: SummaryWriter = None):
        self._print_msg(f'{tag.capitalize()}: {metric}')
        self._write_tb(f'{tag}', metric, writer)

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            SGDActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])
        self._aggregator = StateAggregator()

    def _state(self, cid):
        return self._model.state_dict()

    def _select_client(self):
        return random_select(list(self._fds), s_alpha=self.sample_rate, seed=self.seed + self._k)

    def _local_update_callback(self, cid, res):
        self._aggregator.update(res[0], res[1][0])

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), args)
            for c in cids
        ])):
            self._local_update_callback(cid, res)
            self._metric_averager.update(Metric(*res[1]))

    def _aggregate(self, cids):
        self._model.load_state_dict(
            add(self._model.state_dict(), self._aggregator.compute())
        )
        self._aggregator.reset()

    def _val(self, cids):
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (self._state(cid), self._fds.val(cid), self.batch_size) for cid in cids
        ])):
            self._metric_averager.update(Metric(*res))

    def _test(self):
        if self._k % self.test_step == 0:
            self._pool.submit(
                lambda a, v: a.evaluate.remote(*v),
                (self._state(None), self._fds.test(), self.batch_size)
            )
            self._print_msg('-' * 65)
            self._log_metric(Metric(*self._pool.get_next()), 'test')

    def start(self):
        self._init()
        try:
            while self._k <= self.round:
                # 1.选择参与设备
                selected = self._select_client()
                # 2.本地训练
                self._metric_averager.reset()
                self._local_update(selected)
                self._log_metric(self._metric_averager.compute(), 'train')
                # 3.聚合更新
                self._aggregate(selected)
                # 4.聚合模型验证
                self._metric_averager.reset()
                self._val(selected)
                self._log_metric(self._metric_averager.compute(), 'val')
                # 5.模型测试
                self._test()
                self._update_iter()
        except:
            self._print_msg(traceback.format_exc())
        finally:
            self.close()

    def close(self):
        if self._bar:
            self._bar.close()
        self.writer.close()
        super(FedAvg, self).close()


class ClusteredFL(FedAvg):

    def _init_group(self):
        self._groups = {}

    def _schedule_group(self, cids):
        pass

    def _gid(self, cid):
        for gid in self._groups:
            if cid in self._groups[gid]['clients']:
                return gid
        return None

    def _state(self, cid):
        gid = self._gid(cid)
        if gid is not None:
            return self._groups[gid]['state']
        return self._model.state_dict()

    def _init(self):
        super(ClusteredFL, self)._init()
        self._init_group()
        self._aggregators = {
            gid: StateAggregator()
            for gid in self._groups
        }
        self.writers = {
            gid: SummaryWriter(f'{self.writer.log_dir}/{gid}')
            for gid in self._groups
        }

    def _select_client(self):
        selected = super(ClusteredFL, self)._select_client()
        self._schedule_group(selected)
        return selected

    def _local_update_callback(self, cid, res):
        gid = self._gid(cid)
        self._aggregators[gid].update(res[0], res[1][0])

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), args)
            for c in cids
        ])):
            self._local_update_callback(cid, res)
            self._metric_averager.update(Metric(*res[1]))

    def _aggregate(self, cids):
        for gid in self._aggregators:
            try:
                add_(self._groups[gid]['state'], self._aggregators[gid].compute())
                self._aggregators[gid].reset()
            except AssertionError:
                continue

    def _test(self):
        self._print_msg('-' * 65)
        metrics = []
        for gid in self._groups:
            cs = self._groups[gid]['clients']
            self._print_msg(f"Group {gid}: {len(cs)} clients")
            if len(cs) < 1:
                continue
            self._metric_averager.reset()
            self._val(cs)
            self._log_metric(self._metric_averager.compute(), 'test', self.writers[gid])
            metrics.append(self._metric_averager.compute())
        self._log_metric(average(metrics), 'test')
        metrics.clear()

    def close(self):
        for w in self.writers:
            self.writers[w].close()
        self.writers.clear()
        self._aggregators.clear()
        super(ClusteredFL, self).close()
