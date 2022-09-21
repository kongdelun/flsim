from utils.tool import os_platform

if 'linux' in os_platform():
    BASE = '/home/ncut/Workbase/'
else:
    BASE = 'D:/Project/Python/'

PROJECT = f'{BASE}/fl/'
DATASET = f'{BASE}/dataset/'
TB_OUTPUT = f'{BASE}/output_syn/'
TEST = f'{BASE}/rep/'
