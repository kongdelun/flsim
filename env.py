import logging
from utils.tool import os_platform

if 'linux' in os_platform():
    BASE = '/home/ncut/Workbase'
else:
    BASE = 'D:/Project/Python'


SEED = 2077
PROJECT = f'{BASE}/flsim'
DATASET = f'{BASE}/dataset'
LEAF_ROOT = f'{DATASET}/leaf'
TB_OUTPUT = f'{BASE}/output_syn'
TEST = f'{BASE}/rep'
LOGGING_LEVEL = logging.DEBUG
PRINT_TO_STDOUT = False