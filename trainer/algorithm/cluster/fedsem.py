from copy import deepcopy
from secrets import choice
from typing import Any

import torch
from trainer.algorithm.cluster.base import ClusteredFL
from utils.nn.functional import flatten, add


class FedSem(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedSem, self)._parse_kwargs(**kwargs)
        if sem := kwargs['sem']:
            self.group_num = sem.get('group_num', 2)

    def _init_group(self) -> dict[Any, dict]:
        return {
            i: {'clients': set(), 'state': deepcopy(self._model.state_dict())}
            for i in range(self.group_num)
        }

    def _local_update_args(self, cids):
        self._flush_group(cids)
        return super(FedSem, self)._local_update_args(cids)

    def _local_update_hook(self, cid, res):
        self._cache[cid] = add(self._state(cid), res[0])
        super(FedSem, self)._local_update_hook(cid, res)

    def _flush_group(self, cids):
        new, change = 0., 0.
        for cid in cids:
            old_gid = self._gid(cid)
            if old_gid is None:
                new_gid = choice(list(self._groups.keys()))
                self._groups[new_gid]['clients'].add(cid)
                new += 1
                continue
            else:
                new_gid = self._best_group(cid)
                if new_gid != old_gid:
                    self._groups[old_gid]['clients'].remove(cid)
                    self._groups[new_gid]['clients'].add(cid)
                    change += 1
        self._logger.info("[{}] New: {:.1%}  Change: {:.1%}".format(self._k, new / len(cids), change / len(cids)))

    def _best_group(self, cid):
        idx = torch.argmin(torch.tensor([
            torch.norm(flatten(self._cache[cid]) - flatten(self._groups[gid]['state']), p=2).item()
            for gid in self._groups
        ])).item()
        return list(self._groups.keys())[idx]
