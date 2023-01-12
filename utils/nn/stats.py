import torch
from torch import Tensor
from functools import reduce
from itertools import combinations
from typing import Sequence

import torch.nn.functional as F

def to_numpy(vecs: Sequence[Tensor]):
    return torch.vstack(list(vecs)).detach().cpu().numpy()

def cosine_similarity(vec1: Tensor, vec2: Tensor, eps: float = 1e-08):
    return torch.clip(
        torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + eps),
        -1., 1.
    )


def cosine_dissimilarity(vec1: Tensor, vec2: Tensor, eps: float = 1e-08):
    return (1. - cosine_similarity(vec1, vec2, eps)) / 2.


def kl_divergence(vec1: Tensor, vec2: Tensor):
    return F.kl_div(vec1, vec2)


def js_divergence(vec1: Tensor, vec2: Tensor):
    return (kl_divergence(vec1, vec2) + kl_divergence(vec2, vec1)) / 2.


def l1_diversity(vecs: Sequence[Tensor]):
    return reduce(
        torch.add,
        map(lambda v: torch.norm(v[0] - v[1]), combinations(vecs, 2))
    ) / len(vecs)


def cos_diversity(vecs: Sequence[Tensor]):
    return reduce(
        torch.add,
        map(lambda v: cosine_dissimilarity(v[0], v[1]), combinations(vecs, 2))
    ) / len(vecs)



