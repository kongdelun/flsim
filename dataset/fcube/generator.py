import random
from pathlib import Path

import numpy as np
import torch
from utils.tool import set_seed


def gen_train(num_samples=8000, seed=None):
    set_seed(seed)
    X_train, y_train = [], []
    for loc in range(4):
        for i in range(int(num_samples / 4)):
            p1 = random.random()
            p2 = random.random()
            p3 = random.random()
            if loc > 1:
                p2 = -p2
            if loc % 2 == 1:
                p3 = -p3
            if i % 2 == 0:
                X_train.append([p1, p2, p3])
                y_train.append(0)
            else:
                X_train.append([-p1, -p2, -p3])
                y_train.append(1)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int64)


def gen_test(num_samples=2000, seed=None):
    set_seed(seed)
    X_test, y_test = [], []
    for i in range(num_samples):
        p1 = random.random() * 2 - 1
        p2 = random.random() * 2 - 1
        p3 = random.random() * 2 - 1
        X_test.append([p1, p2, p3])
        if p1 > 0:
            y_test.append(0)
        else:
            y_test.append(1)
    return np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.int64)


def generate_dataset(root: str, num_samples=5000, test_ratio=0.2, seed=2077):
    root = Path(root)
    if not root.exists():
        root.mkdir()
    test_size = int(num_samples * test_ratio)
    train_size = num_samples - test_size
    torch.save(gen_train(train_size, seed), root.joinpath('train.pt'))
    torch.save(gen_test(test_size, seed), root.joinpath('test.pt'))
