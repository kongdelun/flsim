from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset

from benchmark.dataset.leaf import SeqDataset
from benchmark.dataset.shakespeare.preprocess import preprocessing
from utils.data.dataset import FederatedDataset
from utils.select import random_select


class ToVector:
    LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

    def __init__(self, mode=True):
        self.mode = mode

    def __call__(self, x):
        if self.mode:
            return torch.tensor(list(map(lambda ch: self.LETTERS.find(ch), x)))
        else:
            return torch.tensor(self.LETTERS.find(x))


class Shakespeare(FederatedDataset):

    def __init__(self, root: str, sample_rate: Optional[float] = None, seed=None):
        self.root, self.seed = root, seed
        self.users, self.user_data, self.num_samples = self.__load()
        if sample_rate:
            p = np.array([self.num_samples[u] for u in self.users])
            self.users = random_select(self.users, s_alpha=sample_rate, p=p / np.sum(p), seed=seed)

    def __load(self):
        if not Path(self.root, 'processed/index.pt').exists():
            preprocessing(self.root, 0.1)
        raw = torch.load(Path(self.root, 'processed/index.pt'))
        users, user_data, num_samples = raw.pop('users'), raw.pop('user_data'), raw.pop('num_samples')
        return users, user_data, num_samples

    def __contains__(self, key):
        return key in self.users

    def __len__(self):
        return len(self.users)

    def __iter__(self):
        for u in self.users:
            yield u

    def __getitem__(self, key) -> Dataset:
        return ConcatDataset([self.train(key), self.val(key)])

    def train(self, key) -> Dataset:
        datasource = torch.load(self.user_data[key])['train']
        return SeqDataset(datasource, ToVector(), ToVector(False))

    def val(self, key) -> Dataset:
        datasource = torch.load(self.user_data[key])['test']
        return SeqDataset(datasource, ToVector(), ToVector(False))

    @lru_cache(1)
    def test(self) -> Dataset:
        return ConcatDataset([self.val(u) for u in self])
