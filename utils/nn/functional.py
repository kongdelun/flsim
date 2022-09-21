from functools import reduce
from operator import mul
from collections import OrderedDict
from typing import Optional, Sequence

import torch
from torch import Tensor

from benchmark.mnist.model import MLP


def numel(state: dict):
    return sum(map(lambda x: x.numel(), state.values()))


def extract_layer(state: dict, *layer_names):
    layer_names = [ln for ln in layer_names if ln in state]
    sub_state = OrderedDict()
    for ln in layer_names:
        sub_state[ln] = state[ln]
    return sub_state


def extract_shape(state: dict):
    dim = OrderedDict()
    for ln in state:
        dim[ln] = tuple(state[ln].shape)
    return dim


def state2vector(states: Sequence[dict]):
    return list(map(lambda x: flatten(x), states))


def flatten(state: dict):
    return torch.cat(list(map(lambda x: x.flatten(), state.values())))


def unflatten(vector: Tensor, shape: dict):
    layer_size = [reduce(mul, v, 1) for v in shape.values()]
    new_state = OrderedDict()
    for ln, l in zip(shape, torch.split(vector, layer_size)):
        new_state[ln] = torch.reshape(l, shape[ln])
    return new_state


def zero_like(state):
    return sub(state, state)


def add(state1: dict, state2: dict):
    new_state = OrderedDict()
    for ln in state1:
        new_state[ln] = state1[ln] + state2[ln]
    return new_state


def add_(state1: dict, state2: dict):
    for ln in state1:
        state1[ln] += state2[ln]
    return state1


def sub(state1: dict, state2: dict):
    new_state = OrderedDict()
    for ln in state1:
        new_state[ln] = state1[ln] - state2[ln]
    return new_state


def sub_(state1: dict, state2: dict):
    for ln in state1:
        state1[ln] -= state2[ln]
    return state1


def scalar_mul(state: dict, scalar):
    new_state = OrderedDict()
    for ln in state:
        new_state[ln] = state[ln] * scalar
    return new_state


def scalar_mul_(state: dict, scalar):
    for ln in state:
        state[ln] *= scalar
    return state


def linear_sum(states: Sequence[dict], ps: Optional[Sequence] = None):
    if ps is None:
        ps = [1.] * len(states)
    if len(states) - len(ps) < 0:
        ps = ps[:len(states) - len(ps)]
    elif len(states) - len(ps) > 0:
        ps = list(ps) + [1.] * (len(states) - len(ps))
    new_state = OrderedDict()
    for ln in states[0]:
        new_state[ln] = reduce(torch.add, map(lambda st, p: st[ln] * p, states, ps))
    return new_state


def powerball(state: dict, gamma: float):
    new_state = OrderedDict()
    for ln in state:
        new_state[ln] = torch.sign(state[ln]) * torch.pow(torch.abs(state[ln]), gamma)
    return new_state


if __name__ == '__main__':
    s = MLP().state_dict()
    print(numel(s))
