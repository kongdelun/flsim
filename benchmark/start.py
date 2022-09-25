from os import environ
import hydra
from builder import build_model, build_federated_dataset, build_trainer

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['HYDRA_FULL_ERROR'] = '1'
environ['OC_CAUSE'] = '1'


@hydra.main(config_path="", config_name="cfg", version_base=None)
def run(cfg: dict):
    for tn in cfg.trainer.names:
        fds = build_federated_dataset(cfg.dataset.name, cfg.dataset.args)
        net = build_model(cfg.model.name, cfg.model.args)
        trainer = build_trainer(tn, net, fds, cfg.trainer.args)
        trainer.start()


if __name__ == '__main__':
    run()
