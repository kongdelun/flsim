from copy import deepcopy
from random import shuffle
from typing import OrderedDict, Sequence, Optional
from trainer.algorithm.fedavg import FedAvg

from trainer.core.aggregator import Aggregator
from utils.nn.aggregate import shuffle_layer, average
from utils.nn.functional import add_


class MRAggregator(Aggregator):

    def __init__(self, state: OrderedDict, num_parallel: int, syn_layers: Optional[Sequence[str]]):
        super(MRAggregator, self).__init__()
        self._syn_layers = syn_layers if syn_layers else list()
        self._s = [deepcopy(state) for _ in range(num_parallel)]

    def states(self):
        states = deepcopy(self._s)
        shuffle(self._s)
        return states

    def update(self, grad: OrderedDict):
        super(MRAggregator, self).update()
        self._s.append(add_(self._s.pop(0), grad))

    def _compute_step(self):
        # shuffle_layer(self._s)
        state = average(self._s)
        for s in self._s:
            for ln in self._syn_layers:
                s[ln] = state[ln]
        return state


class FedMR(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedMR, self)._parse_kwargs(**kwargs)
        if mr := kwargs['la']:
            self.sync_idx = mr.get('sync_idx', 2)

    def _aggregate(self, cids):
        self._model.load_state_dict(self._aggregator.compute())
        self._aggregator.reset()

    def _build_aggregator(self):
        return MRAggregator(
            self._model.state_dict(),
            int(len(list(self._fds)) * self.sample_rate),
            list(self._model.state_dict().keys())[:self.sync_idx]
        )

    def _local_update_args(self, cids):
        return [(s, self._fds.train(c), self.local_args) for c, s in zip(cids, self._aggregator.states())]

    def _local_update_hook(self, cid, res):
        self._aggregator.update(res[0])
