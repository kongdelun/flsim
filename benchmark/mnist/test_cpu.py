import unittest
import warnings

import ray
from torch import nn, optim
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from trainer.cpu.cfl.cfl import CFL
from trainer.cpu.cfl.fedsem import FedSem
from trainer.cpu.cfl.ifca import IFCA
from trainer.cpu.fedavg import FedAvg

from utils.data.dataset import get_target
from utils.data.partition import Part, BasicPartitioner, client_sample_count, barh_report, get_report
from utils.io import load_yaml

from benchmark.mnist.model import MLP


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter('ignore', ResourceWarning)
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    root = "./class_1/{}"

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        # 读取配置文件
        cfg = load_yaml(self.root.format("cfg.yaml"))
        # 数据集
        dc = cfg['dataset']
        dataset = ConcatDataset([
            MNIST(root=dc['root'], transform=ToTensor(), train=True),
            MNIST(root=dc['root'], transform=ToTensor(), train=False),
        ])
        # 划分器
        partitioner = BasicPartitioner(get_target(dataset), Part.NONIID_LABEL_SKEW, **cfg['partitioner'])
        # 模型
        model = MLP()
        # 本地损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        opt = optim.SGD(model.parameters(), **cfg['opt'])
        # 参数汇总
        self.args = {
            'config': cfg,
            'model': model,
            'criterion': criterion,
            'opt': opt,
            'dataset': dataset,
            'dp': partitioner,
            'verbose': True
        }

    def test_part(self):
        dp = self.args['dp']
        df = get_report(dp.targets, dp.indices, verbose=True)
        df.to_csv(self.root.format('report.csv'))
        barh_report(df, 10, self.root.format('report_10.jpg'))
        client_sample_count(df, self.root.format('csn.jpg'))

    def test_fedavg(self):
        with SummaryWriter(self.root.format('output/fedavg_cpu')) as writer:
            self.args['writer'] = writer
            trainer = FedAvg(**self.args)
            trainer.train()

    def test_fedsem(self):
        with SummaryWriter(self.root.format('output/fedsem')) as writer:
            self.args['writer'] = writer
            trainer = FedSem(**self.args)
            trainer.train()

    def test_ifca(self):
        with SummaryWriter(self.root.format('output/ifca')) as writer:
            self.args['writer'] = writer
            trainer = IFCA(**self.args)
            trainer.train()

    def test_cfl(self):
        with SummaryWriter(self.root.format('output/cfl')) as writer:
            self.args['writer'] = writer
            trainer = CFL(**self.args)
            trainer.train()


if __name__ == '__main__':
    unittest.main()
