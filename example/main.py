from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra

from benchmark.model import CNN32


@hydra.main(version_base=None, config_path="./conf", config_name="opt")
def my_app(cfg: DictConfig):
    print(cfg)
    optim = instantiate(cfg.optimizer)
    CNN32().parameters()


if __name__ == "__main__":
    my_app()
