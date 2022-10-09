import os
import shutil
import subprocess
import sys
import random
from typing import Optional

import torch
import numpy as np
from torch.utils.collect_env import get_platform

units = {
    'b': 1024. ** 0,
    'kb': 1024. ** 1,
    'mb': 1024. ** 2,
    'gb': 1024. ** 3,
    'tb': 1024. ** 4,
}


def sizeof(obj, unit='b'):
    return sys.getsizeof(obj) / units[unit]


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
    print(f"Clear {path}")


def cmd(command: str, timeout: Optional[int] = None, verbose: bool = True):
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )
    result = proc.communicate(timeout=timeout)[0]
    if verbose:
        print(command)
        print(result)
    return result


