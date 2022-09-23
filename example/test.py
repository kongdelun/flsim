import itertools
import math
import time
import unittest

import numpy as np
import torch
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from utils.cache import DiskCache
from utils.data.partition import Part
from utils.result import get_acc_table


def multi_step(x, init_beta, gamma, step):
    return init_beta * (gamma ** (x // step) - 1)


def ln(x, init_beta, step):
    return init_beta * np.log(1 + x // step)


class MyTestCase(unittest.TestCase):
    root = 'D:/project/python/MyFed/{}'

    def test_get_acc_table(self):
        for dn in ['cifar10', 'femnist']:
            tab = get_acc_table(self.root.format('benchmark/{}'.format(dn)), axis=0, verbose=True)
            tab.to_csv(self.root.format('asset/{}.csv'.format(dn)))

    def test_disk_cache(self):
        dc = DiskCache(3, root=f'{self.root}/cache')
        for i in range(10):
            dc[i] = 'nihao'
        del dc[7]
        print(dc)
        print(dc[1], dc)
        print(dc.exists(1))
        dc.delete(1)
        print(dc.exists(1))
        dc['g/3'] = 1
        print(dc[7], dc)
        for k, v in dc.iter():
            print(k, v)

    def test_all(self):
        # s = {1:'23'}
        print(0)
        print(np.argwhere(np.array([1, 0, 1, 0, 1]) == 0).flatten())

    def test_beta(self):
        print(50000 * 1.03 ** 5)
        # x = np.arange(0, 299)
        # # ln = LnMomentum(0.1, 1)
        # ms = MultiStepMomentum(0.2, 1.2, 20)
        # print(ms.step(1))
        # plt.plot(x, ms.step(x))
        # plt.show()
        # plt.plot(x, multi_step(x, 0.3, 1.5, 60))
        # # # plt.plot(x, ln(x, 0.3, 30))

        # # print(ln(10, 0.2, 30))
        # print(np.clip(4, 0., 0.8))

    def test_OmegaConf(self):
        print(Part(1))
        # conf = OmegaConf.create({'C': Part.NONIID_LABEL_SKEW})
        #
        # print(OmegaConf.to_yaml(conf))
