from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Sent140(Dataset):

    @staticmethod
    def load(root):
        df = pd.read_csv(
            root,
            encoding='ISO-8859-1',
            names=['target', 'id', 'date', 'query', 'username', 'content']
        )
        return np.array(df.pop('content')), np.array(df.pop('target'))

    def __init__(self, root, is_train=True, transform=None, target_transform=None):
        if is_train:
            self.__root = str(Path(root).joinpath('trainingandtestdata/training.1600000.processed.noemoticon.csv'))
        else:
            self.__root = str(Path(root).joinpath('trainingandtestdata/testdata.manual.2009.06.14.csv'))
        self.transform = transform
        self.target_transform = target_transform
        self.__data, self.__target = self.load(self.__root)

    def __getitem__(self, index) -> T_co:
        data = self.transform(self.__data[index]) if self.transform else self.__data[index]
        target = self.target_transform(self.__target[index]) if self.target_transform else self.__target[index]
        return data, target

    def __len__(self):
        return self.__target.shape[0]
