import dataclasses
from typing import Iterable

import numpy as np


@dataclasses.dataclass
class Metric:
    num: int
    loss: float
    acc: float

    def __str__(self):
        return 'Loss: {:.3f}  Acc: {:.1%}'.format(self.loss, self.acc)


def average(metrics: Iterable[Metric]):
    nums = list(map(lambda v: v.num, metrics))
    return Metric(
        np.average(nums).item(),
        np.average(list(map(lambda v: v.loss, metrics)), weights=nums).item(),
        np.average(list(map(lambda v: v.acc, metrics)), weights=nums).item()
    )


class MetricAverager:

    def __init__(self):
        self._metrics = []

    def update(self, m: Metric):
        self._metrics.append(m)

    def compute(self):
        return average(self._metrics)

    def reset(self):
        self._metrics.clear()
