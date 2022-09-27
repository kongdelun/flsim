import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from trainer.core.proto import ClusteredFL
from utils.nn.init import with_kaiming_normal


class IFCA(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(IFCA, self)._parse_kwargs(**kwargs)
        if ifca := kwargs['ifca']:
            self.group_num = ifca.get('group_num', 2)

    def _init_group_hook(self):
        for i in range(self.group_num):
            self._groups[i] = {
                'clients': set(),
                'state': with_kaiming_normal(self._model.state_dict())
            }

    def _schedule_group(self, cids):
        # 清空预设
        for gid in self._groups:
            self._groups[gid]['clients'] -= set(cids)
        # 重新安排
        for cid in cids:
            gid = self._best_group(cid)
            self._groups[gid]['clients'].add(cid)

    @torch.no_grad()
    def _best_group(self, cid):
        criterion = CrossEntropyLoss()
        losses, loss = [], MeanMetric()
        self._model.eval()
        for gid in self._groups:
            loss.reset()
            self._model.load_state_dict(self._groups[gid]['state'])
            for data, target in DataLoader(self._fds.train(cid), self.batch_size):
                loss.update(criterion(self._model(data), target))
            losses.append(loss.compute())
        return torch.argmin(torch.stack(losses)).item()
