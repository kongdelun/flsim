from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from benchmark.dataset.fcube.generator import generate_dataset
from utils.data.partition import DataPartitioner
import utils.data.functional as F
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

