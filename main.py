import logging
from os import environ
from pprint import pprint

from omegaconf import OmegaConf
from torch.nn import Module
from benchmark.ctx import synthetic, mnist
from trainer.api import get_trainer
from utils.data.dataset import FederatedDataset
from utils.result import print_text

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.Logger('benchmark').setLevel(logging.INFO)


def run(trainers: list[str], net: Module, fds: FederatedDataset, cfg):
    print_text(OmegaConf.to_yaml(cfg), 5)
    errors = {}
    for tn in trainers:
        try:
            trainer = get_trainer(tn, net, fds, cfg)
            trainer.start()
        except RuntimeError as e:
            errors[tn] = e
            continue

    print_text(f'{len(trainers) - len(errors)} has finished !!!', 1)
    if len(errors) > 0:
        for tn in errors:
            print_text(f'{tn} has errors: {errors[tn]}', 2)


if __name__ == '__main__':
    net, fds, cfg = mnist()
    # cfg['ring']['rho'] = 0.7
    # cfg['tag'] ='_08_7'
    methods = [
        # 'FedAvg', 'FedProx', 'FedAvgM', 'FedDyn','IFCA',
        'FedLA',
        # 'FedSem',
        # 'FedGroup'
        # 'CFL'
        # 'Scaffold'
        # 'Ring'
    ]
    run(methods, net, fds, cfg)
