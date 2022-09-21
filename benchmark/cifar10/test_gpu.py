import unittest
from torch.utils.tensorboard import SummaryWriter

from benchmark.cifar10.config import get_args
from trainer.final.v0.cfl.cfl import CFL
from trainer.final.v0.cfl.fedsem import FedSem
from trainer.final.v0.cfl.ifca import IFCA
from trainer.final.v0.fedavg import FedAvg
from trainer.final.v0.reg.feddyn import FedDyn
from trainer.final.v0.reg.fedprox import FedProx
from trainer.final.v0.seq.fedla import FedLA
from utils.data.partition import Part, get_report, bubble_report, barh_report, client_sample_count
from utils.tool import func_name


class MyTestCase(unittest.TestCase):
    # 调节划分类型
    root = "./class_1/"
    part = Part.NONIID_LABEL_SKEW

    def test_part(self):
        args = get_args(self.root, self.part)
        df = get_report(args['dp'].targets, args['dp'].indices, verbose=True)
        df.to_csv(self.root + 'report.csv')
        bubble_report(df, self.root + 'report_bubble.jpg')
        barh_report(df, 10, self.root + 'report_10.jpg', title="Dir 0.1")
        client_sample_count(df, self.root + 'csn.jpg')

    def test_fedavg(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = FedAvg(**args)
            trainer.train()

    def test_fedprox(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = FedProx(**args)
            trainer.train()

    def test_feddyn(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = FedDyn(**args)
            trainer.train()

    def test_cfl(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = CFL(**args)
            trainer.train()

    def test_fedla(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_5(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 5
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_20(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 20
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_10(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 10
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_eps_3(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 0
            args['config']['da']['eps'] = 0.03
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_eps_5(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 0
            args['config']['da']['eps'] = 0.05
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedla_eps_10(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 0
            args['config']['da']['eps'] = 0.1
            args['config']['da']['beta'] = 0.0
            trainer = FedLA(**args)
            trainer.train()

    def test_fedlam(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = FedLA(**args)
            trainer.train()

    def test_fedlam_2_EX(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 0
            args['config']['da']['eps'] = 0.07
            args['config']['da']['beta'] = 0.2
            trainer = FedLA(**args)
            trainer.train()

    def test_fedlam_EX(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            args['config']['da']['delay_step'] = 0
            trainer = FedLA(**args)
            trainer.train()

    # def test_fedlam_5(self):
    #     log_dir = self.root + 'output/' + func_name().lstrip('test_')
    #     args = get_args(self.root, self.part)
    #     with SummaryWriter(log_dir) as writer:
    #         args['writer'] = writer
    #         args['config']['da']['delay_step'] = 0
    #         args['config']['da']['eps'] = 0.07
    #         args['config']['da']['beta'] = 0.5
    #         trainer = FedLA(**args)
    #         trainer.train()

    def test_fedsem(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = FedSem(**args)
            trainer.train()

    def test_ifca(self):
        log_dir = self.root + 'output/' + func_name().lstrip('test_')
        args = get_args(self.root, self.part)
        with SummaryWriter(log_dir) as writer:
            args['writer'] = writer
            trainer = IFCA(**args)
            trainer.train()

