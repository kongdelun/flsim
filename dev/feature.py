import traceback
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics.pairwise as pw
import torch
from scipy.special import softmax
from scipy.stats import entropy
from torch.nn import CrossEntropyLoss

from trainer.core.proto import FedAvg
from trainer.util.metric import Metric
from utils.data.dataset import sample_by_class
from utils.nn.functional import zero_like, state2vector


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
        self.grads = {
            cid: zero_like(self._model.state_dict())
            for cid in self._fds
        }
        self.auxiliary_sample = sample_by_class(self._fds.test(), 10, 50)

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
                    self.auxiliary_sample)
                ))
                return x

        X = np.vstack([
            loss(self.grads[cid])
            for cid in self.grads
        ])
        self._model.load_state_dict(state)
        return softmax(X)

    def show(self):

        vecs = state2vector(list(self.grads.values()))
        x = torch.stack(vecs).detach().numpy()
        # svd = TruncatedSVD(n_components=10, random_state=2077, algorithm='arpack')
        # decomposed_updates = svd.fit_transform(x.T)
        # x = (1. - pw.cosine_similarity(x, decomposed_updates.T)) / 2.
        # x = self.loss_by_class()

        sns.clustermap(pw.cosine_similarity(x))
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
            self.grads[cid] = res[0]
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
                self._update()
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

    # def __class_loss(self, state, ads):
    #     loss = CrossEntropyLoss
    #     self._model.load_state_dict(state)
    #     self._model.eval()
    #     with torch.no_grad():
    #         losses = list(map(lambda s: loss(self._model(s[0]), s[1]).item(), ads))
    #     return np.array(losses)

    # def class_(self):

    #
    # def decomposed_cosine_dissimilarity(self):
    #     self.__train()
    #     x = np.vstack([flatten(ds).detach().numpy() for ds in self.res.values()])
    #     svd = dec.TruncatedSVD(n_components=10, random_state=2077, algorithm='arpack')
    #     decomposed_updates = svd.fit_transform(x.T)
    #     return (1. - pw.cosine_similarity(x, decomposed_updates.T)) / 2.
    #
    # def grad(self):
    #     self.__train()
    #     x = np.vstack([flatten(s).numpy() for s in self.res.values()])
    #     pca = dec.PCA(10, svd_solver='arpack')
    #     x = pca.fit_transform(x)
    #     return softmax(x)
