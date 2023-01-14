from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torch.utils.data.dataset import T_co, ConcatDataset

from utils.data.partition import DataPartitioner
from utils.io import load_jsons
from utils.tool import os_platform, set_seed


def get_target(dataset: Dataset):
    num_workers, prefetch_factor, batch_size = (2, 4, 4096) if os_platform() == 'linux' else (0, 2, 512)
    return np.concatenate(list(map(
        lambda x: x[-1].numpy(),
        DataLoader(dataset, num_workers=num_workers, prefetch_factor=prefetch_factor, batch_size=batch_size)
    )))


def get_data(dataset: Dataset):
    num_workers, prefetch_factor, batch_size = (2, 4, 2500) if os_platform() == 'linux' else (0, 2, 512)
    return np.concatenate(list(map(
        lambda x: x[0].numpy(),
        DataLoader(dataset, num_workers=num_workers, prefetch_factor=prefetch_factor, batch_size=batch_size)
    )))


def sample_by_class(dataset: Dataset, num_classes, size, seed=None):
    set_seed(seed, use_torch=True)
    samples, counter = {}, 0
    data, target = [], []
    for x, y in DataLoader(dataset, shuffle=True, batch_size=1):
        if samples.get(y.item(), 0) < size:
            data.append(x)
            target.append(y)
            samples[y.item()] = samples.get(y.item(), 0) + 1
            counter += 1
        if counter == num_classes * size:
            break
    return TensorDataset(torch.cat(data), torch.cat(target))


class SampleSubSet(Subset):
    def __init__(self, dataset, ratio=1., seed=None):
        np.random.seed(seed)
        super(SampleSubSet, self).__init__(
            dataset,
            np.random.choice(len(dataset), int(len(dataset) * ratio), replace=False)
        )


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


class FederatedDataset(ABC):

    @abstractmethod
    def __contains__(self, key):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def train(self, key) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def val(self, key) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dataset:
        raise NotImplementedError


class BasicFederatedDataset(FederatedDataset):

    def __init__(self, dataset: Dataset, dp: DataPartitioner):
        self._dp = dp
        self._dataset = dataset

    def __contains__(self, key):
        return key in self._dp

    def __len__(self):
        return len(self._dp)

    def __iter__(self):
        for k in self._dp:
            yield k

    def __getitem__(self, key) -> Dataset:
        return Subset(self._dataset, self._dp[key])

    def train(self, key) -> Dataset:
        return Subset(self._dataset, self._dp.train_indices[key])

    def val(self, key) -> Dataset:
        return Subset(self._dataset, self._dp.val_indices[key])

    def test(self) -> Dataset:
        return Subset(self._dataset, self._dp.test_indices)


class LEAF(FederatedDataset):

    def __init__(self, root, transform=None, target_transform=None):
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
