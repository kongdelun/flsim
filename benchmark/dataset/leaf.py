import sys
import numpy as np
from benchmark.dataset.transformer import ToNumpy, ToVector
from env import LEAF_ROOT
from utils.tool import cmd, quick_clear
from utils.data.dataset import LEAF


class Synthetic(LEAF):

    def __init__(self, root):
        super(Synthetic, self).__init__(
            root,
            transform=ToNumpy(np.float32),
            target_transform=ToNumpy(np.int64)
        )


class Shakespeare(LEAF):

    def __init__(self, root):
        super(Shakespeare, self).__init__(root, ToVector(), ToVector(False))


def leaf_cmd(s, sf=None, k=None, iu=None, tf=None, t=None, seed=None):
    """
    Args:
        s: iid non-iid
        sf: fraction of data to sample, written as a decimal; set it to 1.0 in order to keep the number of tasks/users specified earlier.
        k: minimum number of samples per user; set it to 5.
        iu: number of users, if i.i.d. sampling; expressed as a fraction of the total number of users; default is 0.01
        tf: fraction of data in training set, written as a decimal; default is 0.9.
        t:'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups.
        seed: random split of data and  random sampling of data.
    Returns cmd
    """
    leaf = f"./preprocess.sh -s {s}"
    if sf is not None and 0.0 < sf < 1.0:
        leaf += f" -sf {sf}"
    if k is not None:
        leaf += f" -k {int(k)}"
    if iu is not None and 0.0 < iu < 1.0:
        leaf += f" -iu {iu}"
    if tf is not None and 0.0 < tf < 1.0:
        leaf += f" -tf {tf}"
    if t is not None and t in ['user', 'sample']:
        leaf += f" -t {t}"
    if seed is not None and isinstance(seed, int):
        leaf += f" --smplseed {seed} --spltseed {seed}"
    return leaf


def leaf_clear(dataset, data_dir=None):
    if data_dir is None:
        data_dir = ['rem_user_data', 'sampled_data', 'test', 'train']
    for p in data_dir:
        quick_clear(f"{LEAF_ROOT}/{dataset}/data/{p}")


# (small-sized dataset) ('-tf 0.8' reflects the train-test split used in the FedAvg paper)edavg
# create_shakespeare('niid', sf=0.2, k=0, t='sample', tf=0.8, seed=2077)
def create_shakespeare(s, iu=None, sf=None, k=None, t=None, tf=None, seed=None):
    leaf_clear('shakespeare')
    cmd(f'cd {LEAF_ROOT}/shakespeare && {leaf_cmd(s=s, sf=sf, k=k, iu=iu, tf=tf, t=t, seed=seed)}')


# create_synthetic(100, 5, 60, 'niid', sf=1.0, k=5, t='sample', tf=0.8, seed=2077)
def create_synthetic(num_tasks=100, num_classes=5, num_dim=60, s='iid', sf=None, k=None, t=None, tf=None, seed=None):
    leaf_clear('synthetic', ['rem_user_data', 'sampled_data', 'test', 'train', 'all_data'])
    cmd(f'{sys.executable} {LEAF_ROOT}/synthetic/main.py -num-tasks {num_tasks} -num-classes {num_classes} -num-dim {num_dim}')
    cmd(f'cd {LEAF_ROOT}/synthetic && {leaf_cmd(s=s, sf=sf, k=k, tf=tf, t=t, seed=seed)}')


if __name__ == '__main__':
    pass
