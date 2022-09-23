import enum
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, option_context
from sklearn.model_selection import train_test_split

import utils.data.functional as F
from utils.tool import set_seed


class Part(Enum):
    IID_BALANCE = 0
    IID_UNBALANCE_LOGNORMAL = 1
    IID_UNBALANCE_DIRICHLET = 2
    NONIID_BALANCE_CLIENT_DIRICHLET = 3
    NONIID_UNBALANCE_LOGNORMAL_CLIENT_DIRICHLET = 4
    NONIID_UNBALANCE_DIRICHLET_CLIENT_DIRICHLET = 5
    NONIID_LABEL_SKEW = 6
    NONIID_SHARD = 7
    NONIID_DIRICHLET = 8
    NONIID_SYNTHETIC = 9


class DataPartitioner(ABC):
    """
        Base class for data partition in federated learning.
    """

    def __init__(self):
        self.indices = {}
        self.train_indices = {}
        self.val_indices = {}
        self.test_indices = []

    @abstractmethod
    def _split(self):
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class BasicPartitioner(DataPartitioner):

    def __init__(self, targets, part, client_num, lognormal_sgm=0., dirichlet_alpha=0., dirichlet_min=10,
                 major_class_num=0, shard_num=0, test_ratio=0.2, seed=None):
        super(BasicPartitioner, self).__init__()
        self.targets = targets if isinstance(targets, np.ndarray) else np.array(targets)
        self.sample_num = self.targets.shape[0]
        self.class_num = np.unique(self.targets).shape[0]
        self.part = part
        self.client_num = client_num
        self.seed = seed
        self._lognormal_sgm = lognormal_sgm
        self._dirichlet_min = dirichlet_min
        self._dirichlet_alpha = dirichlet_alpha
        self._major_class_num = major_class_num
        self._shard_num = shard_num
        self.indices |= self._split()
        self._split_indices(test_ratio)

    def _split_indices(self, test_ratio=0.2):
        for i in self.indices:
            t, v = train_test_split(self.indices[i], test_size=test_ratio, random_state=self.seed)
            self.train_indices[i], self.val_indices[i] = t.tolist(), v.tolist()
            self.test_indices.extend(self.val_indices[i])

    def __len__(self):
        return len(self.indices)

    def __contains__(self, index):
        return index in self.indices

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __getitem__(self, index):
        return self.indices[index]

    def _split(self):
        # 切分数据集
        set_seed(self.seed)
        if self.part == Part.IID_BALANCE:
            return F.iid_partition(
                F.balance_split(self.client_num, self.sample_num),
                self.sample_num
            )
        elif self.part == Part.IID_UNBALANCE_LOGNORMAL:
            return F.iid_partition(
                F.unbalance_lognormal_split(self.client_num, self.sample_num, self._lognormal_sgm),
                self.sample_num
            )
        elif self.part == Part.IID_UNBALANCE_DIRICHLET:
            return F.iid_partition(
                F.unbalance_dirichlet_split(self.client_num, self.sample_num, self._dirichlet_alpha,
                                            self._dirichlet_min),
                self.sample_num
            )
        if self.part == Part.NONIID_BALANCE_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.balance_split(self.client_num, self.sample_num),
                self.targets, self.class_num, self._dirichlet_alpha
            )
        elif self.part == Part.NONIID_UNBALANCE_LOGNORMAL_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.unbalance_lognormal_split(self.client_num, self.sample_num, self._lognormal_sgm),
                self.targets, self.class_num, self._dirichlet_alpha
            )
        elif self.part == Part.NONIID_UNBALANCE_DIRICHLET_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.unbalance_dirichlet_split(self.client_num, self.sample_num, self._dirichlet_alpha,
                                            self._dirichlet_min),
                self.targets, self.class_num, self._dirichlet_alpha
            )
        elif self.part == Part.NONIID_LABEL_SKEW:
            # label-distribution-skew:quantity-based
            assert self._major_class_num <= self.class_num, f"major_class_num >= {self.class_num}"
            return F.noniid_label_skew_quantity_based_partition(
                self.targets, self.client_num, self.class_num, self._major_class_num
            )
        elif self.part == Part.NONIID_SHARD:
            # partition is 'shards'
            assert self._shard_num >= self.client_num, f"shard_num >= {self.client_num}"
            return F.noniid_shard_partition(
                self.targets, self.client_num, self._shard_num
            )
        elif self.part == Part.NONIID_DIRICHLET:
            # label-distribution-skew:distributed-based (Dirichlet)
            return F.noniid_dirichlet_partition(
                self.targets, self.client_num, self.class_num, self._dirichlet_alpha, self._dirichlet_min
            )
        else:
            raise NotImplementedError("{} is supported !".format(self.part))


def get_report(targets, client_indices, verbose=False):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    class_num = np.unique(targets).shape[0]
    columns, data = ['client_id', 'sample_num'], []
    columns.extend('class{}'.format(i) for i in range(class_num))
    for c_id in sorted(client_indices.keys()):
        c_targets, c_sample_num = targets[client_indices[c_id]], len(client_indices[c_id])
        c_target_cnt = Counter(c_targets)
        row = [c_id, c_sample_num]
        row.extend([c_target_cnt[i] for i in range(class_num)])
        data.append(row)
    df = DataFrame(data, columns=columns)
    if verbose:
        with option_context('expand_frame_repr', False, 'display.max_rows', None):
            print(df)
    return df


def barh_report(report: DataFrame, n=10, path=None, verbose=True, title=None):
    assert n <= report.shape[0], "n <= {}".format(report.shape[0])
    ax = report.iloc[:n, 2:].plot.barh(stacked=True)
    ax.set_xlabel('Num')
    ax.set_ylabel('Client')
    ax.set_title(title)
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()


def bubble_report(report: DataFrame, path=None, verbose=True):
    T = report.iloc[:, 2:]
    client_num, class_num = T.shape
    x = np.array(list(range(client_num)))
    y = np.array(list(range(class_num)))
    X, Y = np.meshgrid(x, y)
    Z = T.iloc[x, y]
    plt.scatter(X, Y, s=Z, c=X, cmap='viridis')
    plt.xlabel('Client')
    plt.ylabel('Class')
    plt.yticks(y)
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()


# 客户端样本数统计
def client_sample_count(report: DataFrame, path=None, verbose=True):
    data = report.iloc[:, :2]
    sns.histplot(
        data=data,
        x="sample_num",
        edgecolor='none',
        alpha=0.7,
        shrink=0.95,
        color='#4169E1'
    )
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()
