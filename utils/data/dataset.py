from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

from utils.data.partition import DataPartitioner
from utils.tool import os_platform, set_seed


def get_target(dataset: Dataset):
    num_workers, prefetch_factor, batch_size = (2, 4, 2500) if os_platform() == 'linux' else (0, 2, 512)
    return np.concatenate(list(map(
        lambda x: x[-1].numpy(),
        DataLoader(dataset, num_workers=num_workers, prefetch_factor=prefetch_factor, batch_size=batch_size)
    )))


def train_test_indices_split(dp: DataPartitioner, test_ratio=0.2, seed=None):
    train, val, test = {}, {}, []
    for i in range(len(dp)):
        t, v = train_test_split(dp[i], test_size=test_ratio, random_state=seed)
        train[i], val[i] = t.tolist(), v.tolist()
        test.extend(val[i])
    return train, val, test


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

    @abstractmethod
    def secondary(self, *arg, **kwargs) -> Dataset:
        raise NotImplementedError


class BasicFederatedDataset(FederatedDataset):

    def __init__(self, dataset: Dataset, dp: DataPartitioner):
        self._dp = dp
        self._dataset = dataset
        self._secondary = None

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

    def secondary(self, num_classes, size):
        if self._secondary is None:
            self._secondary = sample_by_class(self._dataset, num_classes, size, 2077)
        return self._secondary

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_partitioner(self):
        return self._dp
