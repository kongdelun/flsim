from collections import OrderedDict
from datetime import datetime

import ray
from ray.util import ActorPool
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from trainer.core.actor import CPUActor
from trainer.core.aggregator import Aggregator, NotCalculated
from trainer.core.proto import FedAvg
from utils.cache import DiskCache
from utils.metric import Metric
from utils.nn.aggregate import fedavg
from utils.nn.functional import sub, zero_like, scalar_mul_, add_


@ray.remote
class ScaffoldActor(CPUActor):

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        global_control = args.get('global_control')
        opt = args.get('opt', {'lr': 0.001})
        batch_size = args.get('batch_size', 32)
        epoch = args.get('epoch', 5)
        self.set_state(state)
        local_control = args.get('local_control', None)
        if local_control is None:
            local_control = zero_like(state)
        opt, K, lr = optim.SGD(self.model.parameters(), **opt), 0, opt['lr']
        self.model.train()
        for k in range(epoch):
            for data, target in self.dataloader(dataset, batch_size):
                opt.zero_grad()
                self.loss(self.model(data), target).backward()
                opt.step()
                K += self.__rt(global_control, local_control, lr)
        state_, delta_control = self.get_state(), OrderedDict()
        for ln in local_control:
            delta_control[ln] = - global_control[ln] + (state[ln] - state_[ln]) / (K * lr)
            local_control[ln] += delta_control[ln]
        return sub(state_, state), delta_control, local_control, self.evaluate(state_, dataset, batch_size)

    def __rt(self, global_control: OrderedDict, local_control: OrderedDict, lr):
        for ln in local_control:
            self.get_state()[ln] -= lr * (global_control[ln] - local_control[ln])
        return 1


class ScaffoldAggregator(Aggregator):

    def __init__(self, state: OrderedDict, num_clients, global_lr=1.):
        super(ScaffoldAggregator, self).__init__()
        self._grads = []
        self._delta_controls = []
        self._state = state
        self._control = zero_like(self._state)
        self._num_clients = num_clients
        self._global_lr = global_lr

    def update(self, local_grad: OrderedDict, local_delta_control: OrderedDict):
        super(ScaffoldAggregator, self).update()
        self._grads.append(local_grad)
        self._delta_controls.append(local_delta_control)

    def compute(self):
        try:
            return super(ScaffoldAggregator, self).compute()
        except NotCalculated:
            assert len(self._delta_controls) == len(self._grads) > 0
            delta_control, grad = fedavg(self._delta_controls), fedavg(self._grads)
            add_(self._state, scalar_mul_(grad, self._global_lr))
            add_(self._control, scalar_mul_(delta_control, len(self._grads) * 1. / self._num_clients))
            self._res = self._state
            return self._res

    def reset(self):
        super(ScaffoldAggregator, self).reset()
        self._grads.clear()
        self._delta_controls.clear()

    @property
    def control(self):
        return self._control


class Scaffold(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if scaffold := kwargs['scaffold']:
            self.global_lr = scaffold.get('global_lr', 0.98)

    def _init(self):
        super(FedAvg, self)._init()
        self._pool = ActorPool([
            ScaffoldActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])
        self._aggregator = ScaffoldAggregator(
            self._model.state_dict(),
            len(self._fds), self.global_lr
        )
        self._cache = DiskCache(
            self.cache_size,
            f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        )

    def _local_update_callback(self, cid, res):
        self._aggregator.update(res[0], res[1])
        self._cache[cid] = res[2]

    def _local_update(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'global_control': self._aggregator.control
        }
        for res, cid in zip(self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), dict({'local_control': self._cache.get(c)}, **args))
            for c in cids
        ]), cids):
            self._local_update_callback(cid, res)
            self._metric_averager.update(Metric(*res[3]))

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
