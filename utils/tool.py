import os
import sys
import random
import shutil
import subprocess
import traceback
from typing import Optional, Sequence
from importlib import import_module
import torch
import numpy as np
from torch.utils.collect_env import get_platform
from utils.logger import Logger

_logger = Logger.get_logger(__name__)

def set_seed(seed, use_torch=False, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def func_name():
    return sys._getframe().f_back.f_code.co_name


def force_exit(code):
    return os._exit(code)


def os_platform():
    return get_platform()


def quick_clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    _logger.info(f"Clean {path}")


def cmd(command: str, timeout: Optional[int] = None):
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )
    result = proc.communicate(timeout=timeout)[0]
    _logger.info(command)
    _logger.info(result)
    return result


def locate(modules: Sequence[str], name: str, args: dict):
    for m in modules:
        try:
            return getattr(import_module(m), name)(**args)
        except:
            _logger.debug(traceback.format_exc())
            continue
    raise ImportError(f'The {name} is not found !!!')
