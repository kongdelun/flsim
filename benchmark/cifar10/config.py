from functools import lru_cache

from torch import nn, optim
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from benchmark.cifar10.model import CNN
from utils.data.dataset import get_target
from utils.data.partition import Part, BasicPartitioner
from utils.io import load_yaml


@lru_cache(maxsize=3)
def get_args(root: str, part: Part):
    # 读取配置文件
    cfg = load_yaml(root + 'cfg.yaml')
    dataset = ConcatDataset([
        CIFAR10(
            **cfg['dataset'],
            transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            train=True
        ),
        CIFAR10(
            **cfg['dataset'],
            transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            train=False
        )
    ])
    # 划分器
    partitioner = BasicPartitioner(get_target(dataset), part, **cfg['partitioner'])
    # 模型
    model = CNN()
    # 优化器
    opt = optim.SGD(model.parameters(), **cfg['opt'])
    # 参数汇总
    return {
        'config': cfg,
        'model': model,
        'opt': opt,
        'dataset': dataset,
        'dp': partitioner,
        'verbose': True
    }
