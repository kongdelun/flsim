from typing import Sequence
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class UserDataset(Dataset):

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
