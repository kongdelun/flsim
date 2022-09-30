import logging
import traceback
from os import environ

import hydra
import ray
from omegaconf import DictConfig

from builder import build_model, build_federated_dataset, build_trainer

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['HYDRA_FULL_ERROR'] = '1'
environ['OC_CAUSE'] = '1'


@hydra.main(config_path="", config_name="cfg", version_base=None)
def run(cfg: DictConfig):
    ray.init(include_dashboard=False, log_to_driver=False, logging_level=logging.ERROR)
    fds = build_federated_dataset(cfg.dataset.name, cfg.dataset.args)
    net = build_model(cfg.model.name, cfg.model.args)
    trainer = None
    for tn in cfg.trainer.names:
        try:
            trainer = build_trainer(tn, net, fds, cfg.trainer.args)
            trainer.start()
        except:
            print(traceback.format_exc())
            continue
        finally:
            if trainer is not None:
                trainer.close()


if __name__ == '__main__':
    run()
