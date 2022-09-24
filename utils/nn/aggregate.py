import torch
from typing import Optional, Sequence
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
