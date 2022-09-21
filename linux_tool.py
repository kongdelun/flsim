import os
import shutil
from typing import Iterable

from env import BASE


def base_rm(path):
    path = f"{BASE}/{path}"
    shutil.rmtree(path)
    print(f'{path} has removed')


def ls(path: str):
    for f in os.listdir(f"{BASE}/{path}"):
        print(f)


if __name__ == '__main__':
    # base_rm('/dataset/leaf/data/synthetic/data')
    base_rm('/output_syn/Ring/')
    # ls('')
