import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from benchmark.dataset.leaf import UserDataset
from utils.data.dataset import FederatedDataset, sample_by_class
from utils.io import load_jsons


class ToNumpy:
    def __init__(self, dtype):
        self.__dtype = dtype

    def __call__(self, x):
        return np.array(x, dtype=self.__dtype)


class Synthetic(FederatedDataset):

    def __init__(self, root):
        self._secondary = None
        self.root = root
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
        return UserDataset(self.__train_data['user_data'][key], ToNumpy(np.float32), ToNumpy(np.int64))

    def val(self, key) -> Dataset:
        return UserDataset(self.__test_data['user_data'][key], ToNumpy(np.float32), ToNumpy(np.int64))

    def test(self) -> Dataset:
        return ConcatDataset([self.val(key) for key in self.__users])

    def __contains__(self, key):
        return key in self.__users

    def secondary(self, num_classes, size):
        if self._secondary is None:
            self._secondary = sample_by_class(self.test(), num_classes, size, 2077)
        return self._secondary
