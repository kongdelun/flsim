from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchmetrics import Accuracy

from dataset.fcube.generator import generate_dataset
from utils.data.partition import DataPartitioner
import utils.data.functional as F
from utils.profiler import Timer
from utils.tool import set_seed


class FCUBE(Dataset):
    """FCUBE data set.

    From paper `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        root (str): Root for data file.
        train (bool, optional): Training set or test set. Default as ``True``.
        transform (callable, optional): A function/transform that takes in an ``numpy.ndarray`` and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        num_samples (int, optional): Total number of samples to generate. We suggest to use 4000 for training set, and 1000 for test set. Default is ``4000`` for trainset.
    """

    def __init__(
            self, root, train=True,
            transform=None, target_transform=None,
            num_samples=5000, seed=None
    ):
        self._root = root
        self._train = train
        self._seed = seed
        self._num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self._data, self._target = self._load()

    def _load(self):
        root = Path(self._root)
        if not root.exists():
            generate_dataset(self._root, self._num_samples, seed=self._seed)
        return torch.load(root.joinpath('train.pt' if self._train else 'test.pt'))

    def __getitem__(self, index) -> T_co:
        data = self.transform(self._data[index]) if self.transform else self._data[index]
        target = self.target_transform(self._target[index]) if self.target_transform else self._target[index]
        return data, target

    def __len__(self):
        return self._target.shape[0]


class FCUBEPartitioner(DataPartitioner):
    """FCUBE data partitioner.
    FCUBE is a synthetic dataset for research in non-IID scenario with feature imbalance. This
    dataset and its partition methods are proposed in `Federated Learning on Non-IID Data Silos: An
    Experimental Study <https://arxiv.org/abs/2102.02079>`_.
    Supported partition methods for FCUBE:
    - feature-distribution-skew:synthetic
    - IID
    For more details, please refer to Section (IV-B-b) of original paper.
    Args:
        data (numpy.ndarray): Data of dataset :class:`FCUBE`.
    """

    def __init__(self, data, iid=True, test_ratio=0.2, seed=None):
        super().__init__()
        self.iid = iid
        self.seed = seed
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.indices |= self._split()
        self._split_indices(test_ratio)

    def __getitem__(self, index):
        return self.indices[index]

    def __contains__(self, item):
        return item in self.indices

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self):
        return 4

    def _split(self):
        set_seed(self.seed)
        if self.iid:
            return F.iid_partition(
                F.balance_split(4, self.data.shape[0]),
                self.data.shape[0]
            )
        else:
            # feature-distribution-skew:synthetic
            return F.noniid_fcube_synthetic_partition(self.data)

    def _split_indices(self, test_ratio=0.2):
        for i in self.indices:
            t, v = train_test_split(self.indices[i], test_size=test_ratio, random_state=self.seed)
            self.train_indices[i], self.val_indices[i] = t.tolist(), v.tolist()
            self.test_indices.extend(self.val_indices[i])
#
#
# net = nn.Sequential(
#     nn.Linear(3, 9),
#     nn.ReLU(),
#     nn.Dropout(),
#     nn.Linear(9, 2),
# ).cuda()
#
# ds = FCUBE('./raw', True)
# acc = Accuracy().cuda()
# opt = optim.Adam(net.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss().cuda()
# for _ in range(1000):
#     with Timer("cost"):
#         net.train()
#         for x, y in DataLoader(ds, batch_size=32):
#             x, y = x.cuda(), y.cuda()
#             opt.zero_grad()
#             logit = net(x)
#             loss = loss_fn(logit, y)
#             acc.update(logit, y)
#             loss.backward()
#             opt.step()
#         print(acc.compute())

