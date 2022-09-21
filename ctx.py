from torch import nn
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from benchmark.mnist.model import MLP
from dataset.fcube.fcube import FCUBE
from dataset.shakespeare.shakespeare import Shakespeare
from dataset.synthetic import Synthetic
from env import DATASET, PROJECT
from utils.data.dataset import BasicFederatedDataset, get_target, FederatedDataset, sample_by_class
from utils.data.partition import BasicPartitioner, Part
from utils.io import load_yaml

cfg = load_yaml(f'{PROJECT}/trainer/cfg.yaml')


def mnist():
    ds = ConcatDataset([
        MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=True),
        MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=False),
    ])

    # 划分器
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = MLP()
    return net, fds, cfg['trainer']


def synthetic():
    fds = Synthetic(f"{DATASET}/leaf/data/synthetic/data/")
    net = nn.Sequential(
        nn.Linear(60, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    conf = cfg['trainer']
    conf['cfl']['eps_1'] = 0.03
    conf['cfl']['eps_2'] = 0.4
    return net, fds, conf


def fcude():
    fds = ConcatDataset([
        FCUBE(f'{DATASET}/fcube/raw', True, num_samples=60000),
        FCUBE(f'{DATASET}/fcube/raw', False)
    ])
    net = nn.Sequential()


def shakespeare():
    fds = Shakespeare(f'{DATASET}/shakespeare/raw')
    net = nn.Sequential(
        nn.Linear(3, 2),
        nn.PReLU()
    )



