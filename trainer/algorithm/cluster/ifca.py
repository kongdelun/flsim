from copy import deepcopy
from random import choice
from typing import Any

from trainer.algorithm.cluster.base import ClusteredFL


class IFCA(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(IFCA, self)._parse_kwargs(**kwargs)
        if ifca := kwargs['ifca']:
            self.group_num = ifca.get('group_num', 2)

    def _init_group(self) -> dict[Any, dict]:
        return {
            i: {'clients': set(), 'state': deepcopy(self._model.state_dict())}
            for i in range(self.group_num)
        }

    def _local_update_args(self, cids):
        self._flush_group(cids)
        return super(IFCA, self)._local_update_args(cids)

    def _flush_group(self, cids):
        new, change = 0., 0.
        for cid in cids:
            old_gid = self._gid(cid)
            if old_gid is None:
                new_gid = choice(list(self._groups.keys()))
                self._groups[new_gid]['clients'].add(cid)
                new += 1
                continue
            new_gid = self._best_group(cid)
            if new_gid != old_gid:
                self._groups[old_gid]['clients'].remove(cid)
                self._groups[new_gid]['clients'].add(cid)
                change += 1
        self._logger.info("[{}] New: {:.1%}  Change: {:.1%}".format(self._k, new / len(cids), change / len(cids)))

    def _best_group(self, cid):
        best_gid = None
        min_loss = 1000
        for gid, res in zip(self._groups, self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (self._groups[gid]['state'], self._fds.train(cid), self.batch_size)
            for gid in self._groups
        ])):
            if res[1] < min_loss:
                min_loss = res[1]
                best_gid = gid
        return best_gid


