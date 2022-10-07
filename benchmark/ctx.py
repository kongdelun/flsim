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
from utils.logger import Logger
from utils.nlp import ToFixedSeq
from utils.tool import func_name

logger = Logger.get_logger(__name__)


def shakespeare(**kwargs):
    kwargs.clear()
    root = f'{LEAF_ROOT}/{func_name()}/data/'
    fds = Shakespeare(root)
    return fds


def synthetic(**kwargs):
    kwargs.clear()
    root = f'{LEAF_ROOT}/{func_name()}/data/'
    fds = Synthetic(root)
    return fds


# mnist
def mnist(partitioner, **kwargs):
    kwargs.clear()
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        MNIST(root, transform=ToTensor(), train=True),
        MNIST(root, transform=ToTensor(), train=False),
    ])
    # 划分器
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def femnist(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    split = kwargs.get('split', 'letters')
    ds = ConcatDataset([
        EMNIST(root, split=split, train=True, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split)),
        EMNIST(root, split=split, train=False, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split))
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def fmnist(partitioner, **kwargs):
    kwargs.clear()
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FashionMNIST(root, train=True, transform=ToTensor()),
        FashionMNIST(root, train=False, transform=ToTensor())
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def fcube(partitioner, **kwargs):
    train_size = kwargs.get('train_size', 50000)
    test_size = kwargs.get('test_size', 10000)
    ds = ConcatDataset([
        FCUBE(train_size=train_size, seed=partitioner['seed']),
        FCUBE(test_size=test_size, seed=partitioner['seed'])
    ])
    fds = FCUBEPartitioner(get_data(ds), **partitioner)
    return fds


def sent140(partitioner, **kwargs):
    root = f'{DATASET}/{func_name()}/raw/'
    transform = ToFixedSeq(kwargs.get('max_len', 35), kwargs.get('dim', 25))
    target_transform = tf.ToSent140Target()
    ds = SampleSubSet(
        ConcatDataset([
            Sent140(root, True, transform, target_transform),
            Sent140(root, False, transform, target_transform)
        ]), kwargs.get('sample_ratio', 0.1), seed=partitioner['seed']
    )
    logger.info(f'Using {len(ds)} samples from Sent140')
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def cifar10(partitioner, **kwargs):
    kwargs.clear()
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        CIFAR10(root, train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        CIFAR10(root, train=False, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    ])
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds


def celeba(partitioner, **kwargs):
    kwargs.clear()
    root = f'{DATASET}/{func_name()}/raw/'
    ds = CelebA(
        root, split='all',
        target_type='attr',
        transform=Compose([Resize(32), ToTensor()]),
        target_transform=tf.ToCelebaAttrTarget(1)
    )
    dp = BasicPartitioner(get_target(ds), **partitioner)
    fds = BasicFederatedDataset(ds, dp)
    return fds
