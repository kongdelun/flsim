import traceback
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F

from benchmark.ctx import synthetic
from trainer.core.proto import FedAvg
from utils.metric import Metric
from utils.cache import DiskCache
from utils.nn.functional import add


# import torch
# from ray.util import ActorPool
# import sklearn.decomposition as dec
# from scipy.special import softmax
# from torch.nn import Module, CrossEntropyLoss
#
# from root import TEST
# from utils.data.dataset import sample_by_class, FederatedDataset, BasicFederatedDataset
# from utils.io import save_dict, load_dict
# from utils.select import random_select
# from trainer.core.actor import SGDActor


class Representation(FedAvg):

    def _init(self):
        super(Representation, self)._init()
        self._k = -1
        self.secondary = self._fds.secondary(10, 20)
        self._cache = DiskCache(
            self.cache_size,
            f'{self.writer.log_dir}/run/{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}'
        )
        # self.grads = {
        #     cid: zero_like(self._model.state_dict())
        #     for cid in self._fds
        # }

    def loss_by_class(self):
        criterion = CrossEntropyLoss()
        state = deepcopy(self._state(None))

        # loss func
        def loss(st):
            self._model.load_state_dict(st)
            self._model.eval()
            with torch.no_grad():
                x = np.array(list(map(
                    lambda s: criterion(self._model(s[0]), s[1]).item(),
                    self.secondary)
                ))
                return x

        X = np.vstack([
            loss(self.grads[cid])
            for cid in self.grads
        ])
        self._model.load_state_dict(state)
        return softmax(X)

    @torch.no_grad()
    def logit(self, t=1.):
        self._model.eval()
        for cid in self._fds:
            self._model.load_state_dict(self._cache[cid]['state'])
            for data, target in DataLoader(self.secondary, batch_size=10 * 20):
                self._cache[cid]['logit'] = self._model(data) / t

    def show(self, ):
        kl_dist = np.zeros((len(self._fds), len(self._fds)))
        self.logit()
        for i, c1 in enumerate(self._fds):
            for j, c2 in enumerate(self._fds):
                kl_dist[i][j] = F.kl_div(self._cache[c1]['logit'], self._cache[c2]['logit']).numpy()
        print(kl_dist.shape)

        # vecs = state2vector(list(self.grads.values()))
        # x = torch.stack(vecs).detach().numpy()
        # svd = TruncatedSVD(n_components=10, random_state=2077, algorithm='arpack')
        # decomposed_updates = svd.fit_transform(x.T)
        # x = (1. - pw.cosine_similarity(x, decomposed_updates.T)) / 2.
        # x = self.loss_by_class()
        sns.clustermap(kl_dist)
        plt.show()

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
            self._cache[cid] = {
                'state': add(self._state(cid), res[0]),
                'grad': res[0],
                'num_sample': res[1][0]
            }
            self._aggregator.update(res[0], res[1][0])
            yield Metric(*res[1])

    def _test(self):
        self._pool.submit(
            lambda a, v: a.evaluate.remote(*v),
            (self._state(None), self._fds.test(), self.batch_size)
        )
        m = Metric(*self._pool.get_next())
        self._print_msg(f'Test: {m}')

    def start(self):
        self._init()
        try:
            while self._k < self.round:
                self._update_iter()
                selected = self._select_client()
                for m in self._local_update(selected):
                    self._metric_averager.update(m)
                m = self._metric_averager.compute()
                self._print_msg(f'Train: {m}')
                self._metric_averager.reset()
                self._aggregate(selected)
                if self._k % self.test_step == 0:
                    self._test()
                    self.show()
        except:
            self._print_msg(traceback.format_exc())
        finally:
            self.close()


if __name__ == '__main__':
    net, fds, cfg = synthetic()

    cfg['round'] = 1
    cfg['sample_rate'] = 1.
    rep = Representation(net, fds, **cfg)
    rep.start()
