from collections import OrderedDict

import ray
from ray.util import ActorPool
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from trainer.core.actor import CPUActor
from trainer.core.aggregator import Aggregator
from trainer.core.proto import FedAvg
from utils.nn.aggregate import average
from utils.nn.functional import sub, zero_like, scalar_mul_, add_


@ray.remote
class ScaffoldActor(CPUActor):

    def _setup(self, args: dict):
        self._batch_size = args.get('batch_size', 32)
        self._epoch = args.get('epoch', 5)
        self._max_grad_norm = args.get('max_grad_norm', 10.0)
        self._opt = args.get('opt', {'lr': 0.001})
        self._global_control = args.get('global_control')
        self._local_control = args.get('local_control', None)
        if self._local_control is None:
            self._local_control = zero_like(self._global_control)

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self._setup(args)
        self.set_state(state)
        opt, K, lr = optim.SGD(self.model.parameters(), **self._opt), 0, self._opt['lr']
        self.model.train()
        for k in range(self._epoch):
            for data, target in self.dataloader(dataset, self._batch_size):
                opt.zero_grad()
                self.loss(self.model(data), target).backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                opt.step()
                K += self.__rt(lr)
        state_, delta_control = self.get_state(), OrderedDict()
        for ln in self._local_control:
            delta_control[ln] = - self._global_control[ln] + (state[ln] - state_[ln]) / (K * lr)
            self._local_control[ln] += delta_control[ln]
        return sub(state_, state), self.evaluate(state_, dataset, self._batch_size), delta_control, self._local_control

    def __rt(self, lr):
        for ln in self._local_control:
            self.get_state()[ln] -= lr * (self._global_control[ln] - self._local_control[ln])
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

    def reset(self):
        super(ScaffoldAggregator, self).reset()
        self._grads.clear()
        self._delta_controls.clear()

    @property
    def control(self):
        return self._control

    def _adapt_fn(self):
        assert len(self._delta_controls) == len(self._grads) > 0
        delta_control, grad = average(self._delta_controls), average(self._grads)
        add_(self._state, scalar_mul_(grad, self._global_lr))
        add_(self._control, scalar_mul_(delta_control, len(self._grads) * 1. / self._num_clients))
        return self._state


class Scaffold(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        if scaffold := kwargs['scaffold']:
            self.global_lr = scaffold.get('global_lr', 0.98)

    def _configure_actor_pool(self):
        self._pool = ActorPool([
            ScaffoldActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])

    def _configure_aggregator(self):
        self._aggregator = ScaffoldAggregator(
            self._model.state_dict(),
            len(self._fds), self.global_lr
        )

    def _local_update_hook(self, cid, res):
        self._aggregator.update(res[0], res[2])
        self._cache[cid] = res[3]

    def _local_update_setup(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'global_control': self._aggregator.control,
            'max_grad_norm': self.max_grad_norm
        }
        return [(self._state(c), self._fds.train(c), dict({'local_control': self._cache.get(c)}, **args)) for c in cids]

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()
