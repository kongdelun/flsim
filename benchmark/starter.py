import sys
from os import environ
from traceback import format_exc

import hydra
from omegaconf import DictConfig
from torchinfo import summary
from benchmark.fds import build_federated_dataset
from benchmark.model import build_model
from env import TB_OUTPUT
from trainer.core.trainer import build_trainer
from utils.logger import Logger

_logger = Logger.get_logger(__name__)


def setup_env():
    sys.argv.append(f'hydra.run.dir={TB_OUTPUT}')
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    environ['HYDRA_FULL_ERROR'] = '1'
    environ['OC_CAUSE'] = '1'


@hydra.main(config_path="config", config_name="cfg", version_base=None)
def run(cfg: DictConfig):
    fds = build_federated_dataset(cfg.dataset.name, cfg.dataset.fds)
    net = build_model(cfg.model.name, cfg.model.args)
    summary(net)
    for tn in cfg.trainer.names:
        try:
            trainer = build_trainer(tn, net, fds, cfg.trainer.args)
            trainer.start()
        except:
            _logger.info(format_exc())


if __name__ == '__main__':
    setup_env()
    run()
