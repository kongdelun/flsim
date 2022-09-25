from torch import nn
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CelebA, CIFAR10, EMNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from benchmark.dataset.emnist import ToTarget
from benchmark.dataset.fcube.fcube import FCUBE
from benchmark.dataset.sent140 import Sent140
from benchmark.dataset.shakespeare.shakespeare import Shakespeare
from benchmark.dataset.synthetic import Synthetic
from benchmark.partitioner.fcube import FCUBEPartitioner
from env import DATASET
from utils.data.dataset import BasicFederatedDataset, get_target, get_data
from utils.data.partition import BasicPartitioner
from utils.nlp import ToFixedSeq
from utils.tool import func_name


def mnist(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        MNIST(root, transform=ToTensor(), train=True),
        MNIST(root, transform=ToTensor(), train=False),
    ])
    # 划分器
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def synthetic(**kwargs):
    root = f'{DATASET}/leaf/data/{func_name()}/data/'
    fds = Synthetic(root)
    return fds


def fcube(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FCUBE(root, True, num_samples=60000),
        FCUBE(root, False, num_samples=20000)
    ])
    fds = FCUBEPartitioner(get_data(ds), **partitioner)
    return fds


def shakespeare(**kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    fds = Shakespeare(root)
    return fds


def sent140(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        Sent140(root, is_train=True, transform=ToFixedSeq(35, 25)),
        Sent140(root, is_train=False, transform=ToFixedSeq(35, 25)),
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def celeba(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = CelebA(root, split='all', download=False, transform=Compose([Resize(32), ToTensor()]))
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def cifar10(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        CIFAR10(root, train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        CIFAR10(root, train=False, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def femnist(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    split = 'letters'
    ds = ConcatDataset([
        EMNIST(root, split=split, train=True, transform=ToTensor(), target_transform=ToTarget(split)),
        EMNIST(root, split=split, train=False, transform=ToTensor(), target_transform=ToTarget(split))
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def fmnist(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FashionMNIST(root, train=True, transform=ToTensor(), target_transform=ToTensor()),
        FashionMNIST(root, train=False, transform=ToTensor(), target_transform=ToTensor())
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds
