from abc import abstractmethod
from datetime import datetime
from typing import Any, Sequence

from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from env import TB_OUTPUT
from trainer.core.actor import BasicActor
from trainer.core.aggregator import BasicAggregator
from trainer.core.trainer import FLTrainer
from utils.cache import DiskCache
from trainer.utils.metric import MetricAverager, Metric
from utils.nn.functional import add_
from utils.nn.init import with_kaiming_normal
from utils.select import random_select


class ClusteredFL(FLTrainer):

    def _gid(self, cid):
        for gid in self._groups:
            if cid in self._groups[gid]['clients']:
                return gid

    def _state(self, cid):
        gid = self._gid(cid)
        if gid is not None:
            return self._groups[gid]['state']
        return self._model.state_dict()

    def _init(self):
        super(ClusteredFL, self)._init()
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
        self._aggregator = self._build_global_aggregator()
        self._pool = self._build_actor_pool()
        self._metric_averager = MetricAverager()
        self._groups = self._init_group()
        for gid in self._groups:
            self._groups[gid]['writer'] = SummaryWriter(f'{self._writer.log_dir}/{gid}')
            self._groups[gid]['aggregator'] = self._build_group_aggregator()

    def _select_client(self):
        return random_select(list(self._fds), s_alpha=self.sample_rate, seed=self.seed + self._k)

    def _local_update(self, cids):
        self._metric_averager.reset()
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), self._local_update_args(cids))):
            self._local_update_hook(cid, res)
            self._metric_averager.update(Metric(*res[1]))
        self._handle_metric(self._metric_averager.compute(), 'train', self._writer)

    def _aggregate(self, cids):
        for gid in self._groups:
            try:
                agg = self._groups[gid]['aggregator']
                add_(self._groups[gid]['state'], agg.compute())
                agg.reset()
                self._aggregator.update(self._groups[gid]['state'], 1)
            except AssertionError:
                continue
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()

    def _eval(self, cids):
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (self._state(cid), self._fds.val(cid), self.batch_size) for cid in cids
        ])):
            self._metric_averager.update(Metric(*res))

    def _val(self, cids):
        self._metric_averager.reset()
        self._eval(cids)
        self._handle_metric(self._metric_averager.compute(), 'val', self._writer)

    def _group_test(self, gid):
        cs = self._groups[gid]['clients']
        self._logger.info(f"[{self._k}] Group({gid}): {sorted(cs)}")
        if len(cs) > 0:
            self._metric_averager.reset()
            self._eval(cs)
            self._handle_metric(self._metric_averager.compute(), 'test', self._groups[gid]['writer'])

    def _global_test(self):
        self._metric_averager.reset()
        self._pool.submit(lambda a, v: a.evaluate.remote(*v), (
            self._state(None), self._fds.test(), self.batch_size
        ))
        self._metric_averager.update(Metric(*self._pool.get_next()))
        self._handle_metric(self._metric_averager.compute(), 'test', self._writer)

    def _test(self):
        for gid in self._groups:
            self._group_test(gid)
        self._global_test()

    def _clean(self):
        for gid in self._groups:
            self._groups[gid]['writer'].close()
            self._groups[gid]['aggregator'].reset()
            self._groups[gid].pop('writer')
            self._groups[gid].pop('aggregator')
        self._writer.close()
        self._aggregator.reset()
        self._metric_averager.reset()
        super(ClusteredFL, self)._clean()

    def _build_actor_pool(self):
        return ActorPool([
            BasicActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])

    def _build_group_aggregator(self):
        return BasicAggregator()

    def _build_global_aggregator(self):
        return BasicAggregator()

    def _local_update_args(self, cids):
        return [(self._state(c), self._fds.train(c), self.local_args) for c in cids]

    def _local_update_hook(self, cid, res):
        gid = self._gid(cid)
        if gid is None:
            self._logger.warn(f"[{self._k}] The client({cid}) has no valid group id.")
            return
        self._groups[gid]['aggregator'].update(res[0], res[1][0])

    @abstractmethod
    def _init_group(self) -> dict[Any, dict]:
        raise NotImplementedError


def grouping(cids: Sequence, labels: Sequence):
    assert len(cids) == len(labels)
    res = {}
    for c, l in zip(cids, labels):
        if l not in res:
            res[l] = []
        res[l].append(c)
    return res
