from omegaconf import OmegaConf
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
from utils.nlp import ToFixedSeq
from utils.tool import func_name

cfg = OmegaConf.load(f"{PROJECT}/benchmark/cfg.yaml")


def mnist():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        MNIST(cfg.root, transform=ToTensor(), train=True),
        MNIST(cfg.root, transform=ToTensor(), train=False),
    ])
    # 划分器
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = cv.MLP()
    return net, fds, cfg.trainer


def synthetic():
    fds = Synthetic(f"{DATASET}/leaf/data/synthetic/data/")
    net = nn.Sequential(
        nn.Linear(60, 90),
        nn.ReLU(),
        nn.Linear(90, 10)
    )
    cfg.trainer.cfl.eps_1 = 0.03
    cfg.trainer.cfl.eps_2 = 0.4
    return net, fds, cfg.trainer


def fcube():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FCUBE(cfg.root, True, num_samples=60000),
        FCUBE(cfg.root, False)
    ])
    fds = FCUBEPartitioner(get_data(ds), Part(cfg.partitioner.pop('part')), cfg.partitioner.seed)
    net = nn.Sequential(
        nn.Linear(3, 9),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(9, 2)
    )
    return net, fds, cfg.trainer


def shakespeare():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    fds = Shakespeare(cfg.root)
    net = nlp.RNN()
    return net, fds, cfg.trainer


def sent140():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        Sent140(cfg.root, is_train=True,
                transform=ToFixedSeq(f'{DATASET}/glove/', 35, 25)),
        Sent140(cfg.root, is_train=False,
                transform=ToFixedSeq(f'{DATASET}/glove/', 35, 25)),
    ])
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = nlp.MLP()
    return net, fds, cfg.trainer


def celeba():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = CelebA(cfg.root, split='all', download=False,
                transform=Compose([Resize(32), ToTensor()]))
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN32()
    return net, fds, cfg.trainer


def cifar10():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        CIFAR10(cfg.root, train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        CIFAR10(cfg.root, train=False, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    ])
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN32()
    return net, fds, cfg.trainer


def femnist():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        EMNIST(cfg.root, split='letters', train=True,
               transform=ToTensor(), target_transform=ToTarget('letters')),
        EMNIST(cfg.root, split='letters', train=False,
               transform=ToTensor(), target_transform=ToTarget('letters'))
    ])
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN21()
    return net, fds, cfg.trainer


def fmnist():
    cfg.root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FashionMNIST(cfg.root, train=True,
                     transform=ToTensor(), target_transform=ToTensor()),
        FashionMNIST(cfg.root, train=False,
                     transform=ToTensor(), target_transform=ToTensor())
    ])
    dp = BasicPartitioner(get_target(ds), Part(cfg.partitioner.pop('part')), **cfg.partitioner)
    fds = BasicFederatedDataset(ds, dp)
    net = cv.CNN21()
    return net, fds, cfg.trainer
