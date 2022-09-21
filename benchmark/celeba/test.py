import unittest
import warnings

import ray
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Compose, Resize

from benchmark.celeba.model import CNN
from benchmark.celeba.partition import CELEBAPartitioner
from trainer.cpu.cfl.fedsem import FedSem
from trainer.cpu.cfl.ifca import IFCA
from trainer.cpu.cfl.cfl import CFL
from trainer.cpu.fedavg import FedAvg
from utils.data.dataset import get_target
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
        self.config = load_yaml("cfg.yaml")
        print(self.config)
        # 数据集
        dc = self.config['dataset']
        self.dataset = CelebA(root=dc['root'], split='all',
                              transform=Compose([Resize(32), ToTensor()]),
                              download=False)
        # 划分器
        pc = self.config['partitioner']
        self.partitioner = CELEBAPartitioner(get_target(self.dataset), **pc)
        # 模型
        mc = self.config['model']
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), mc['lr'])

    def test_fedavg(self):
        with SummaryWriter("./output/fedavg") as writer:
            trainer = FedAvg(
                self.config, self.model, self.criterion, self.opt,
                self.dataset, self.partitioner,
                writer, verbose=True
            )
            trainer.train()

    def test_fedsem(self):
        with SummaryWriter("./output/fedsem") as writer:
            trainer = FedSem(
                self.config, self.model, self.criterion, self.opt,
                self.dataset, self.partitioner,
                writer, verbose=True)
            trainer.train()

    def test_ifca(self):
        with SummaryWriter("./output/ifca") as writer:
            trainer = IFCA(
                self.config, self.model, self.criterion, self.opt,
                self.dataset, self.partitioner,
                writer, verbose=True
            )
            trainer.train()

    def test_cfl(self):
        with SummaryWriter("./output/cfl") as writer:
            trainer = CFL(
                self.config, self.model, self.criterion, self.opt,
                self.dataset, self.partitioner,
                writer, verbose=True
            )
            trainer.train()


if __name__ == '__main__':
    unittest.main()
