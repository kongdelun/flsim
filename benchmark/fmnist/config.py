from functools import lru_cache
from torch import nn, optim
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from benchmark.fmnist.model import CNN
from utils.data.dataset import get_target
from utils.data.partition import Part, BasicPartitioner
from utils.io import load_yaml


@lru_cache(maxsize=3)
def get_args(root: str, part: Part):
    # 读取配置文件
    cfg = load_yaml(root + 'cfg.yaml')
    # 数据集
    dataset = ConcatDataset([
        FashionMNIST(**cfg['dataset'], transform=ToTensor(), train=True),
        FashionMNIST(**cfg['dataset'], transform=ToTensor(), train=False),
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
