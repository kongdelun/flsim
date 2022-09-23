from frozenlist import FrozenList

from trainer.algorithm.cfl.cfl import CFL
from utils.metric import Metric
from utils.compressor.basic import TopkCompressor
from utils.nn.functional import linear_sum, add_, zero_like
from utils.select import random_select


# new method
class RingCFL(CFL):

    def _parse_kwargs(self, **kwargs):
        super(RingCFL, self)._parse_kwargs(**kwargs)
        if ring := kwargs['ring']:
            self.rho = ring.get('rho', 0.7)
            self.compress_ratio = ring.get('compress_ratio', 0.6)

    def _init_group(self):
        super(RingCFL, self)._init_group()
        self._cur = -1
        self.selected_num = int(self.sample_rate * len(self._fds))
        self._compressor = TopkCompressor(self.compress_ratio)
        self._mom = zero_like(self._model.state_dict())

    def _select_client(self):
        self._cur = (self._cur + 1) % len(self._groups)
        return random_select(
            FrozenList(self._groups[self._cur]['clients']),
            self._k,
            s_num=self.selected_num,
            seed=self.seed + self._k
        )

    def _compress(self, state):
        values, indices = self._compressor.compress(state)
        return self._compressor.decompress(values, indices, state)

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
            self._cache[cid] = self._compress(res[0])
            gid = self._gid(cid)
            self._aggregators[gid].update(self._cache[cid], res[1][0])
            yield Metric(*res[1])

    def _aggregate(self, cids):
        grad = self._aggregators[self._cur].compute()
        self._mom = linear_sum([self._mom, grad], [self.rho, 1.])
        add_(self._groups[self._cur]['state'], self._mom)
        self._aggregators[self._cur].reset()

    # def _schedule_group(self, cids):
    #     group_num = len(self._groups)
    #     super(RingCFL, self)._schedule_group(cids)
    #     if group_num < len(self._groups):
    #         self._mom = zero_like(self._model.state_dict())