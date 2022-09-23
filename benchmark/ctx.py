from torch import nn
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CelebA, CIFAR10, EMNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

from benchmark.dataset.emnist import ToTarget
from benchmark.dataset.fcube.fcube import FCUBE
from benchmark.dataset.sent140 import Sent140
from benchmark.dataset.shakespeare.shakespeare import Shakespeare
from benchmark.dataset.synthetic import Synthetic
from benchmark.model import nlp, cv
from benchmark.partitioner.fcube import FCUBEPartitioner
from env import DATASET, PROJECT
from utils.data.dataset import BasicFederatedDataset, get_target, get_data
from utils.data.partition import BasicPartitioner, Part
from utils.io import load_yaml
from utils.nlp import ToFixedSeq

cfg = load_yaml(f'{PROJECT}/benchmark/cfg.yaml')


def mnist():
    ds = ConcatDataset([
        MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=True),
        MNIST(f'{DATASET}/mnist/raw/', transform=ToTensor(), train=False),
    ])

    # 划分器
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = cv.MLP()
    return net, fds, cfg['trainer']


def synthetic():
    fds = Synthetic(f"{DATASET}/leaf/data/synthetic/data/")
    net = nn.Sequential(
        nn.Linear(60, 90),
        nn.ReLU(),
        nn.Linear(90, 10)
    )
    conf = cfg['trainer']
    conf['cfl']['eps_1'] = 0.03
    conf['cfl']['eps_2'] = 0.4
    return net, fds, conf


def fcube():
    ds = ConcatDataset([
        FCUBE(f'{DATASET}/fcube/raw', True, num_samples=60000),
        FCUBE(f'{DATASET}/fcube/raw', False)
    ])
    fds = FCUBEPartitioner(get_data(ds), Part.NONIID_SYNTHETIC, cfg['partitioner']['seed'])
    net = nn.Sequential(
        nn.Linear(3, 9),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(9, 2)
    )
    return net, fds, cfg['trainer']


def shakespeare():
    fds = Shakespeare(f'{DATASET}/shakespeare/raw')
    net = nlp.RNN()
    return net, fds, cfg['trainer']


def sent140():
    ds = ConcatDataset([
        Sent140(root=f'{DATASET}/sent140/raw/', is_train=True,
                transform=ToFixedSeq(f'{DATASET}/glove/', 35, 25)),
        Sent140(root=f'{DATASET}/sent140/raw/', is_train=False,
                transform=ToFixedSeq(f'{DATASET}/glove/', 35, 25)),
    ])
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = nlp.MLP()
    return net, fds, cfg['trainer']


def celeba():
    ds = CelebA(
        root=f'{DATASET}/celeba/raw/', split='all',
        transform=Compose([Resize(32), ToTensor()]),
        download=False
    )
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN32()
    return net, fds, cfg['trainer']


def cifar10():
    ds = ConcatDataset([
        CIFAR10(
            root=f'{DATASET}/cifar10/raw/', train=True,
            transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ),
        CIFAR10(
            root=f'{DATASET}/cifar10/raw/', train=False,
            transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
    ])
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN32()
    return net, fds, cfg['trainer']


def femnist():
    ds = ConcatDataset([
        EMNIST(f'{DATASET}/emnist/raw/', split='letters',
               transform=ToTensor(), target_transform=ToTarget('letters'),
               train=True),
        EMNIST(f'{DATASET}/emnist/raw/', split='letters',
               transform=ToTensor(), target_transform=ToTarget('letters'),
               train=False)
    ])
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN21()
    return net, fds, cfg['trainer']


def fmnist():
    ds = ConcatDataset([
        FashionMNIST(f'{DATASET}/fmnist/raw/',
                     transform=ToTensor(), target_transform=ToTensor(),
                     train=True),
        FashionMNIST(f'{DATASET}/fmnist/raw/',
                     transform=ToTensor(), target_transform=ToTensor(),
                     train=False)
    ])
    dp = BasicPartitioner(get_target(ds), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN21()
    return net, fds, cfg['trainer']
