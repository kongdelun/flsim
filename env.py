import logging
from utils.tool import os_platform


PRINT_TO_STDOUT = True
SEED = 2077
LOGGING_LEVEL = logging.DEBUG


BASE = '/home/ncut/Workbase' if 'linux' in os_platform() else 'D:/Project/Python'

PROJECT = f'{BASE}/flsim'
DATASET = f'{BASE}/dataset'
LEAF_ROOT = f'{DATASET}/leaf'

TEST = f'{BASE}/rep'
TB_OUTPUT = f'{BASE}/output_syn'

