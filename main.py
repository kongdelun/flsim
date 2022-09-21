from os import environ
from pprint import pprint
from torch.nn import Module
from ctx import synthetic, mnist
from trainer.api import get_trainer
from utils.data.dataset import FederatedDataset
from utils.result import print_text

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run(trainers: list[str], net: Module, fds: FederatedDataset, cfg: dict):
    pprint(cfg, indent=4)
    errors = {}
    for tn in trainers:
        try:
            trainer = get_trainer(tn, net, fds, cfg)
            trainer.start()
        except RuntimeError as e:
            errors[tn] = e
            continue

    print_text(f'{len(trainers) - len(errors)} has finished !!!', 2)
    if len(errors) > 0:
        for tn in errors:
            print_text(f'{tn} has errors: {errors[tn]}', 3)


if __name__ == '__main__':
    net, fds, cfg = synthetic()
    # cfg['ring']['rho'] = 0.7
    # cfg['tag'] ='_08_7'
    run(['FedSem', 'IFCA', 'FedGroup'], net, fds, cfg)
