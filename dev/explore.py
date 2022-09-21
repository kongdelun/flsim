import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from benchmark.mnist.model import MLP
from feature import Representation
from env import DATASET

from utils.data.dataset import BasicFederatedDataset, get_target
from utils.data.partition import BasicPartitioner, Part
from utils.io import load_yaml
from utils.nn.stats import cosine

cfg = load_yaml('cfg.yaml')
ds = ConcatDataset([
    MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=True),
    MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=False),
])
# 划分器
dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
fds = BasicFederatedDataset(ds, dp)
net = MLP()

# rep = Representation(net, fds, **cfg['trainer'], tag='mnist')
# rep.start()


# m = rep.class_()
# # sns.heatmap(m)
# # m = rep.decomposed_cosine_dissimilarity()
# # sns.heatmap(m)
# # m = rep.grad()
# sns.heatmap(m)
# plt.show()
# sns.heatmap()
a = torch.randn(100)
print(cosine(a, a))


# LEAF Dataset
# def get_leaf_dataset():




# class Synthetic(Dataset):
#     def __init__(self, n_samples=100, n_features=10, n_classes=2, n_redundant=0, n_informative=2,
#                  n_clusters_per_class=2, random_state=42):
#         super().__init__()
#         from sklearn.datasets import make_classification
#         self.data, self.target = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
#                                                      n_redundant=n_redundant, n_informative=n_informative,
#                                                      n_clusters_per_class=n_clusters_per_class,
#                                                      random_state=random_state)
#         self.data = torch.from_numpy(self.data).float()
#         self.target = torch.from_numpy(self.target).long()
#
#     def __getitem__(self, item):
#         return self.data[item], self.target[item]
#
#     def __len__(self):
#         return len(self.data)
