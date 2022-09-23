import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ray.util import ActorPool
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
import sklearn.metrics.pairwise as pw
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from benchmark.src.mnist.model import MLP
from root import DATASET
from trainer.core.actor import SGDActor
from utils.data.dataset import sample_by_class, FederatedDataset, BasicFederatedDataset, get_target
from utils.data.partition import BasicPartitioner, Part
from utils.io import load_yaml
from utils.nn.functional import flatten
from utils.select import random_select


def train(model: Module, fds: FederatedDataset, **kwargs):
    pool = ActorPool([
        SGDActor.remote(model, CrossEntropyLoss())
        for _ in range(kwargs.get('actor_num', 5))
    ])
    cs = random_select(list(fds), s_alpha=kwargs.get('sample_rate', 1.), seed=2077)
    args = {
        'opt': kwargs.get('opt', {'lr': 0.01}),
        'batch_size': kwargs.get('batch_size', 32),
        'epoch': kwargs.get('epoch', 20)
    }
    for cid, res in zip(cs, pool.map(lambda a, v: a.fit.remote(*v), [
        (model.state_dict(), fds.train(c), args)
        for c in cs
    ])):
        yield cid, res


class Representation:

    def __init__(self, model, fds: BasicFederatedDataset):
        self.loss = CrossEntropyLoss()
        self._fds = fds
        self.model = model
        self.ads = sample_by_class(self._fds.dataset, 10, 20)
        self.res = {}

    def train(self):
        if len(self.res) > 0:
            return
        for c, r in train(self.model, self._fds):
            self.res[c] = r[0]

    def class_loss(self, state):
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            losses = list(map(lambda s: self.loss(self.model(s[0]), s[1]).item(), self.ads))
        return np.array(losses)

    def class_ratio(self):
        self.train()
        tmp = []
        for g in self.res.values():
            x = self.class_loss(g)
            x[x < 0.] = 0.
            tmp.append(softmax(x))
        return np.vstack(tmp)

    def decomposed_cosine_dissimilarity(self):
        self.train()
        x = np.vstack([flatten(ds).detach().numpy() for ds in self.res.values()])
        svd = TruncatedSVD(n_components=5, random_state=2077, algorithm='arpack')
        decomposed_updates = svd.fit_transform(x.T)
        return (1. - pw.cosine_similarity(x, decomposed_updates.T)) / 2.

    def lp_distance(self):
        self.train()
        x = np.vstack([flatten(s).numpy() for s in self.res.values()])

        # # fa = SparseRandomProjection()
        # x = fa.fit_transform(x)
        return x


cfg = load_yaml('./cfg.yaml')
ds = ConcatDataset([
    MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=True),
    MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=False),
])
# 划分器
dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
fds = BasicFederatedDataset(ds, dp)
net = MLP()

rep = Representation(net, fds)
# features =
m = rep.class_ratio()
sns.clustermap(data=pw.cosine_similarity(m))
m = rep.decomposed_cosine_dissimilarity()
sns.clustermap(data=pw.cosine_similarity(m))
m = rep.lp_distance()
sns.clustermap(data=pw.cosine_similarity(m))
plt.show()
