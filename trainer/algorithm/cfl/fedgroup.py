from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
from sklearn.cluster import KMeans, k_means
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from trainer.core.proto import ClusteredFL
from utils.cache import DiskCache
from utils.nn.aggregate import fedavg
from utils.nn.functional import flatten, add
from utils.nn.stats import cosine
from utils.select import random_select


class FedGroup(ClusteredFL):

    def _parse_kwargs(self, **kwargs):
        super(FedGroup, self)._parse_kwargs(**kwargs)
        if group := kwargs['group']:
            self.group_num = group.get('group_num', 2)
            self.pre_ratio = group.get('pre_ratio', 0.3)
            self.pre_epoch = group.get('pre_epoch', 30)
        self.pre_state = deepcopy(self._model.state_dict())

    def _init_group(self):
        super(FedGroup, self)._init_group()
        self._cache = DiskCache(
            self.cache_size,
            f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        )
        cids = random_select(self._fds, s_alpha=self.pre_ratio)
        self._pretrain(cids)
        nums = [self._cache[cid]['num_sample'] for cid in cids]
        grads = [self._cache[cid]['pre_grad'] for cid in cids]
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
                'state': add(self.pre_state, pre_grad),
                'clients': set(cs)
            }

    def _pretrain(self, cids):
        args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.pre_epoch
        }
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self.pre_state, self._fds.train(cid), args)
            for cid in cids
        ])):
            self._cache[cid] = {
                'pre_grad': res[0],
                'num_sample': res[1][0]
            }

    def _has_group(self, cid):
        for gid in self._groups:
            if cid in self._groups[gid]['clients']:
                return True
        return False

    def _schedule_group(self, cids):
        ucs = list(filter(lambda x: not self._has_group(x), cids))
        self._pretrain(ucs)
        for cid in ucs:
            gid = self._best_group(cid)
            self._groups[gid]['clients'].add(cid)

    def _best_group(self, cid):
        return torch.argmin(torch.tensor([
            1. - cosine(flatten(self._cache[cid]['pre_grad']), flatten(self._groups[gid]['pre_grad'])).item()
            for gid in self._groups
        ])).item()
