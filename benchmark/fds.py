import functools

from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CelebA, CIFAR10, EMNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import benchmark.dataset.transformer as tf
from benchmark.dataset.fcube import FCUBE
from benchmark.dataset.leaf import Shakespeare, Synthetic
from benchmark.dataset.sent140 import Sent140
from benchmark.partitioner.fcube import FCUBEPartitioner
from env import DATASET, LEAF_ROOT
from utils.data.dataset import BasicFederatedDataset, get_target, get_data, SampleSubSet
from utils.data.partition import BasicPartitioner
from utils.nlp import ToFixedSeq
from utils.tool import func_name, locate


def build_federated_dataset(name: str, args: dict):
    args = dict(args)
    return locate([
        f'benchmark.fds'
    ], name, args)


def shakespeare(args: dict):
    root = f'{LEAF_ROOT}/{func_name()}/data/'
    fds = Shakespeare(root)
    return fds


def synthetic(args: dict):
    root = f'{LEAF_ROOT}/{func_name()}/data/'
    fds = Synthetic(root)
    return fds


# mnist
def mnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        MNIST(root, transform=ToTensor(), train=True),
        MNIST(root, transform=ToTensor(), train=False),
    ])
    # 划分器
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def femnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    split = args.get('split', 'letters')
    ds = ConcatDataset([
        EMNIST(root, split=split, train=True, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split)),
        EMNIST(root, split=split, train=False, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split))
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def fmnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FashionMNIST(root, train=True, transform=ToTensor()),
        FashionMNIST(root, train=False, transform=ToTensor())
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def fcube(args: dict):
    train_size = args.get('train_size', 50000)
    test_size = args.get('test_size', 10000)
    seed = args.get('seed', 2077)
    ds = ConcatDataset([
        FCUBE(train_size=train_size, seed=seed),
        FCUBE(test_size=test_size, seed=seed)
    ])
    fds = FCUBEPartitioner(get_data(ds), args)
    return fds


def sent140(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    seed = args.get('seed', 2077)
    transform = ToFixedSeq(args.get('max_len', 35), args.get('dim', 25))
    target_transform = tf.ToSent140Target()
    ds = SampleSubSet(ConcatDataset([
        Sent140(root, True, transform, target_transform),
        Sent140(root, False, transform, target_transform)
    ]), args.get('sample_ratio', 0.1), seed=seed)
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def cifar10(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        CIFAR10(root, train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        CIFAR10(root, train=False, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def celeba(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = CelebA(
        root, split='all',
        target_type='attr',
        transform=Compose([Resize(32), ToTensor()]),
        target_transform=tf.ToCelebaAttrTarget(1)
    )
    dp = BasicPartitioner(get_target(ds), args)
    fds = BasicFederatedDataset(ds, dp)
    return fds
