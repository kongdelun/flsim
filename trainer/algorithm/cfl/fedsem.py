import random

import torch
from trainer.core.proto import ClusteredFL
from utils.nn.functional import flatten, add
from utils.nn.init import with_kaiming_normal


class FedSem(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedSem, self)._parse_kwargs(**kwargs)
        if sem := kwargs['sem']:
            self.group_num = sem.get('group_num', 2)

    def _init_group_hook(self):
        for i in range(self.group_num):
            self._groups[i] = {'clients': set(), 'state': with_kaiming_normal(self._model.state_dict())}

    def _gid(self, cid):
        gid = super(FedSem, self)._gid(cid)
        if gid is None:
            gid = random.choice(list(self._groups.keys()))
            # gid = np.argmin([len(self._groups[gid]['clients']) for gid in self._groups])
            self._groups[gid]['clients'].add(cid)
        return gid

    def _local_update_hook(self, cid, res):
        self._cache[cid] = {
            'state': add(self._state(cid), res[0]),
            'num_sample': res[1][0]
        }

    def _best_group(self, cid):
        return torch.argmin(torch.tensor([
            torch.norm(flatten(self._cache[cid]['state']) - flatten(self._groups[gid]['state']), p=2).item()
            for gid in self._groups
        ])).item()

    def _aggregate(self, cids):
        # 清空原有安排
        for gid in self._groups:
            self._groups[gid]['clients'] -= set(cids)
        for cid in cids:
            gid = self._best_group(cid)
            self._groups[gid]['clients'].add(cid)
            self._aggregators[gid].update(**self._cache[cid])
        for gid in self._aggregators:
            try:
                self._groups[gid]['state'] = self._aggregators[gid].compute()
                self._aggregators[gid].reset()
            except AssertionError:
                continue
