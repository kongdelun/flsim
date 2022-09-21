import unittest
import warnings

import ray
from torch import nn, optim
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

from dataset.emnist import ToTarget
from torchvision.datasets import EMNIST
from benchmark.femnist.model import CNN
from trainer.cpu.cfl.cfl import CFL

from trainer.cpu.cfl.fedsem import FedSem
from trainer.cpu.cfl.ifca import IFCA
from trainer.cpu.fedavg import FedAvg

from utils.data.dataset import get_target
from utils.data.partition import Part, BasicPartitioner
from utils.io import load_yaml


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter('ignore', ResourceWarning)
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        # 读取配置文件
        cfg = load_yaml("cfg.yaml")
        # 数据集
        dc = cfg['dataset']
        split = dc['__split']
        dataset = ConcatDataset([
            EMNIST(root=dc['root'], split=split, transform=ToTensor(), target_transform=ToTarget(split), train=True),
            EMNIST(root=dc['root'], split=split, transform=ToTensor(), target_transform=ToTarget(split), train=False),
        ])
        # 划分器
        partitioner = BasicPartitioner(get_target(dataset), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
        # 模型
        model = CNN()
        # 本地损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        opt = optim.SGD(model.parameters(), **cfg['opt'])
        # 参数汇总
        self.args = {
            'config': cfg,
            'model': model,
            '_criterion': criterion,
            'opt': opt,
            'dataset': dataset,
            'dp': partitioner,
            'verbose': True
        }

    def test_fedavg(self):
        with SummaryWriter("./output/fedavg") as writer:
            self.args['writer'] = writer
            trainer = FedAvg(**self.args)
            trainer.train()

    def test_fedsem(self):
        with SummaryWriter("./output/fedsem") as writer:
            self.args['writer'] = writer
            trainer = FedSem(**self.args)
            trainer.train()

    def test_ifca(self):
        with SummaryWriter("./output/ifca") as writer:
            self.args['writer'] = writer
            trainer = IFCA(**self.args)
            trainer.train()

    def test_mycfl(self):
        with SummaryWriter("./output/cfl") as writer:
            self.args['writer'] = writer
            trainer = CFL(**self.args)
            trainer.train()


if __name__ == '__main__':
    unittest.main()
