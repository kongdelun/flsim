import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from utils.tool import set_seed


def gen_train(num_samples):
    X_train, y_train = [], []
    for loc in range(4):
        for i in range(int(num_samples / 4)):
            p1 = random.random()
            p2 = random.random()
            p3 = random.random()
            if loc > 1:
                p2 = -p2
            if loc % 2 == 1:
                p3 = -p3
            if i % 2 == 0:
                X_train.append([p1, p2, p3])
                y_train.append(0)
            else:
                X_train.append([-p1, -p2, -p3])
                y_train.append(1)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64)


def gen_test(num_samples):
    X_test, y_test = [], []
    for i in range(num_samples):
        p1 = random.random() * 2 - 1
        p2 = random.random() * 2 - 1
        p3 = random.random() * 2 - 1
        X_test.append([p1, p2, p3])
        if p1 > 0:
            y_test.append(0)
        else:
            y_test.append(1)
    return np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.int64)


def gen_data(train_size=None, test_size=None, seed=None):
    set_seed(seed)
    if train_size is not None:
        return gen_train(train_size)
    elif test_size is not None:
        return gen_train(test_size)
    else:
        raise ValueError('train_size or test_size must be specified')


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
            self, train_size=None, test_size=None,
            transform=None, target_transform=None,
            seed=None
    ):
        self.transform = transform
        self.target_transform = target_transform
        self._data, self._target = gen_data(train_size, test_size, seed)

    def __getitem__(self, index) -> T_co:
        data = self.transform(self._data[index]) if self.transform else self._data[index]
        target = self.target_transform(self._target[index]) if self.target_transform else self._target[index]
        return data, target

    def __len__(self):
        return self._target.shape[0]
