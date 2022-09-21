from math import ceil

import numpy as np
from typing import Sequence
from utils.tool import set_seed


def extract_sequence(seq: Sequence, indices: Sequence = None):
    if indices is None:
        return seq
    return [e for i, e in enumerate(seq) if i in indices]


def random_select(seq: Sequence, s_alpha=1., s_num=None, p=None, seed=None, is_sorted=False):
    set_seed(seed)
    c_num = len(seq)
    s_num = c_num if s_num is None else s_num
    select_num = min(c_num, s_num, ceil(c_num * s_alpha))
    selected = extract_sequence(seq, np.random.choice(c_num, select_num, False, p))
    # 返回值
    if is_sorted:
        selected.sort()
    return selected


def ribbon_select(seq: Sequence, idx, s_alpha=1., s_num=None, p=None, seed=None, is_sorted=False):
    c_num = len(seq)
    s_num = s_num if s_num else c_num
    select_num = min(c_num, s_num, int(c_num * s_alpha))
    set_seed(seed)
    if idx < c_num / select_num:
        b = idx * select_num
        e = b + select_num if b + select_num < c_num else c_num
        selected = seq[b:e]
        if len(selected) < select_num:
            selected += extract_sequence(seq[:b], np.random.choice(b, select_num - len(selected), False, p))
    else:
        selected = extract_sequence(seq, np.random.choice(c_num, select_num, False, p))
    # 返回值
    if is_sorted:
        selected.sort()
    return selected


def get_prob(nks: Sequence[int]):
    nks = np.array(nks)
    return nks / np.sum(nks)


if __name__ == '__main__':
    pro = get_prob(list(range(50)))
    print(np.random.choice(50, 4, p=pro))
