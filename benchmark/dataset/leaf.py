from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, ConcatDataset
from utils.data.dataset import FederatedDataset, sample_by_class
from utils.io import load_jsons


class ToNumpy:
    def __init__(self, dtype):
        self.__dtype = dtype

    def __call__(self, x):
        return np.array(x, dtype=self.__dtype)


class ToVector:

    LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

    def __init__(self, mode=True):
        self.mode = mode

    def __call__(self, x):
        if self.mode:
            return torch.tensor(list(map(lambda ch: self.LETTERS.find(ch), x)))
        else:
            return torch.tensor(self.LETTERS.find(x))


class SeqDataset(Dataset):

    def __init__(self, datasource: Sequence, transform=None, target_transform=None):
        self.datasource = datasource
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if isinstance(self.datasource, dict):
            return len(self.datasource['y'])
        return len(self.datasource)

    def __getitem__(self, index) -> T_co:
        if isinstance(self.datasource, dict):
            data, target = self.datasource['x'][index], self.datasource['y'][index]
        else:
            data, target = self.datasource[index]
        data = self.transform(data) if self.transform else data
        target = self.target_transform(target) if self.target_transform else target
        return data, target


class LEAF(FederatedDataset):

    def __init__(self, root, transform=None, target_transform=None):
        self._secondary = None
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.__train_data = self.__load('train')
        self.__test_data = self.__load('test')
        self.__users = self.__train_data['users']

    def __load(self, tag='train'):
        data = {}
        for js in load_jsons(f"{self.root}/{tag}/"):
            data.update(js)
        return data

    def __len__(self):
        return len(self.__users)

    def __iter__(self):
        for user in self.__users:
            yield user

    def __getitem__(self, key) -> Dataset:
        return ConcatDataset([self.train(key), self.val(key)])

    def train(self, key) -> Dataset:
        return SeqDataset(self.__train_data['user_data'][key], self.transform, self.target_transform)

    def val(self, key) -> Dataset:
        return SeqDataset(self.__test_data['user_data'][key], self.transform, self.target_transform)

    def test(self) -> Dataset:
        return ConcatDataset([self.val(key) for key in self.__users])

    def __contains__(self, key):
        return key in self.__users

    def secondary(self, num_classes, size):
        if self._secondary is None:
            self._secondary = sample_by_class(self.test(), num_classes, size, 2077)
        return self._secondary
