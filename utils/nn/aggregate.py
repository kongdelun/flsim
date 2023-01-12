from random import shuffle
from typing import Optional, Sequence

import torch
from utils.nn.functional import linear_sum, flatten, unflatten, extract_shape


def average(states: Sequence[dict], weights: Optional[Sequence] = None):
    if weights is None:
        weights = torch.ones(len(states))
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)
    return linear_sum(states, weights / torch.sum(weights))


def pc_grad(states: Sequence[dict]):
    def _proj(state: dict):
        fs = flatten(state)
        for os in states:
            fos = flatten(os)
            proj_direct = torch.dot(fs, fos) / torch.norm(fos) ** 2
            fs -= min(proj_direct, 0.) * fos
        return unflatten(fs, extract_shape(state))

    if states is None or len(states):
        raise ValueError(states)
    return list(map(_proj, states))


def proj(state1: dict, state2: dict):
    fs = flatten(state1)
    fs_ = flatten(state2)
    proj_direct = torch.dot(fs, fs_) / torch.norm(fs_) ** 2
    fs -= min(proj_direct, 0.) * fs_
    return unflatten(fs, extract_shape(state1))


def shuffle_layer(states: Sequence[dict], layer_names: Optional[Sequence[str]] = None, seed=None):
    assert len(states) > 0
    if layer_names is None:
        layer_names = [ln for ln in states[0]]
    for ln in layer_names:
        lws = [s[ln] for s in states]
        shuffle(lws, seed)
        for s, lw in zip(states, lws):
            s[ln] = lw
