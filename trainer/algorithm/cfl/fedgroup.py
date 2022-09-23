from copy import deepcopy
from typing import OrderedDict

import numpy as np
import torch
from sklearn.cluster import k_means
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from trainer.core.proto import ClusteredFL
from utils.nn.aggregate import fedavg
from utils.nn.functional import flatten, add
from utils.nn.stats import cosine
from utils.select import random_select


class FedGroup(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedGroup, self)._parse_kwargs(**kwargs)
        if group := kwargs['group']:
            self.group_num = group.get('group_num', 2)
            self.pre_settings = group.get('pre_settings', {
                'ratio': 0.3, 'epoch': 30, 'lr': 0.01,
            })
            self.pre_settings['state'] = deepcopy(self._model.state_dict())

    def _init_group(self):
        super(FedGroup, self)._init_group()
        # self._cache = DiskCache(
        #     self.cache_size,
        #     f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        # )
        cids = random_select(self._fds, s_alpha=self.pre_settings['ratio'])
        nums, grads = [], []
        for cid, n, g in self._pretrain(cids):
            nums.append(n)
            grads.append(g)

        X = np.vstack(list(map(lambda x: flatten(x).detach().numpy(), grads)))
        svd = TruncatedSVD(self.group_num, algorithm='arpack', random_state=self.seed)
        decomposed_grads = svd.fit_transform(X.T)
        M = (1. - cosine_similarity(X, decomposed_grads.T)) / 2.
        _, labels, _ = k_means(M, self.group_num, random_state=self.seed)
        for gid in range(self.group_num):
            pre_grad = fedavg(
                [g for g, l in zip(grads, labels) if l == gid],
                [n for n, l in zip(nums, labels) if l == gid]
            )
            cs = [c for c, l in zip(cids, labels) if l == gid]
            self._groups[gid] = {
                'pre_grad': pre_grad,
                'state': add(self.pre_settings['state'], pre_grad),
                'clients': set(cs)
            }

    def _pretrain(self, cids):
        args = {
            'opt': {'lr': self.pre_settings['lr']},
            'batch_size': self.batch_size,
            'epoch': self.pre_settings['epoch'],
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self.pre_settings['state'], self._fds.train(c), args)
            for c in cids
        ])):
            yield cid, res[1][0], res[0]

    def _schedule_group(self, cids):
        ucs = list(filter(lambda x: self._gid(x) is None, cids))
        for cid, n, g in self._pretrain(ucs):
            gid = self._best_group(g)
            self._groups[gid]['clients'].add(cid)

    def _best_group(self, grad: OrderedDict):
        return torch.argmin(torch.tensor([
            1. - cosine(flatten(grad), flatten(self._groups[gid]['pre_grad'])).item()
            for gid in self._groups
        ])).item()
