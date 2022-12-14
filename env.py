import logging
from utils.tool import os_platform

OS_ROOT = {
    'linux': '/home/ncut/Workbase',
    'win32': 'D:/Project/Python'
}

BASE = OS_ROOT[os_platform()]

PRINT_TO_STDOUT = False
SEED = 2077
LOGGING_LEVEL = logging.DEBUG

PROJECT = f'{BASE}/flsim'
DATASET = f'{BASE}/dataset'
LEAF_ROOT = f'{DATASET}/leaf'

TEST = f'{BASE}/rep'
TB_OUTPUT = f'{BASE}/output_syn_300'
