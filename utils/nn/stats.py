import torch
from torch import Tensor
from functools import reduce
from itertools import combinations
from typing import Sequence


def cosine(vec1: Tensor, vec2: Tensor, eps: float = 1e-08):
    return torch.clip(
        torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + eps),
        -1., 1.
    )


def diff(vecs: Sequence[Tensor]):
    return reduce(
        torch.add,
        map(lambda v: torch.norm(v[0] - v[1]), combinations(vecs, 2))
    ) / len(vecs)


def cosine_diff(vecs: Sequence[Tensor]):
    return reduce(
        torch.add,
        map(lambda v: .5 * (1 - cosine(v[0], v[1])), combinations(vecs, 2))
    ) / len(vecs)
