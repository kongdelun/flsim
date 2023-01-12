from datetime import datetime
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from env import TB_OUTPUT
from trainer.core.actor import BasicActor
from trainer.core.aggregator import BasicAggregator
from trainer.core.trainer import FLTrainer
from trainer.utils.metric import MetricAverager, Metric
from utils.cache import DiskCache
from utils.nn.functional import add
from utils.nn.init import with_kaiming_normal
from utils.select import random_select


class FedAvg(FLTrainer):

    def _state(self, cid):
        return self._model.state_dict()

    def _init(self):
        super(FedAvg, self)._init()
        self._model.load_state_dict(
            with_kaiming_normal(self._model.state_dict())
        )
        self._writer = SummaryWriter(
            f'{TB_OUTPUT}/{self.name}'
        )
        self._cache = DiskCache(
            self.cache_size,
            f'{self._writer.log_dir}/run/{datetime.today().strftime("%H-%M-%S")}'
        )
        self._aggregator = self._build_aggregator()
        self._pool = self._build_actor_pool()
        self._metric_averager = MetricAverager()

    def _select_client(self):
        return random_select(list(self._fds), s_alpha=self.sample_rate, seed=self.seed + self._k)

    def _local_update(self, cids):
        self._metric_averager.reset()
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), self._local_update_args(cids))):
            self._local_update_hook(cid, res)
            self._metric_averager.update(Metric(*res[1]))
        self._handle_metric(self._metric_averager.compute(), 'train', self._writer)

    def _aggregate(self, cids):
        self._model.load_state_dict(
            add(self._model.state_dict(), self._aggregator.compute())
        )
        self._aggregator.reset()

    def _val(self, cids):
        self._metric_averager.reset()
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (self._state(cid), self._fds.val(cid), self.batch_size) for cid in cids
        ])):
            self._metric_averager.update(Metric(*res))
        self._handle_metric(self._metric_averager.compute(), 'val', self._writer)

    def _test(self):
        self._metric_averager.reset()
        self._pool.submit(lambda a, v: a.evaluate.remote(*v), (
            self._state(None), self._fds.test(), self.batch_size
        ))
        self._metric_averager.update(Metric(*self._pool.get_next()))
        self._handle_metric(self._metric_averager.compute(), 'test', self._writer)

    def _clean(self):
        self._writer.close()
        self._aggregator.reset()
        self._metric_averager.reset()
        super(FedAvg, self)._clean()

    def _build_actor_pool(self):
        return ActorPool([
            BasicActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])

    def _build_aggregator(self):
        return BasicAggregator()

    def _local_update_args(self, cids):
        return [(self._state(c), self._fds.train(c), self.local_args) for c in cids]

    def _local_update_hook(self, cid, res):
        self._aggregator.update(res[0], res[1][0])
