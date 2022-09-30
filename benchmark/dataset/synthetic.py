import numpy as np
from benchmark.dataset.leaf import LEAF, ToNumpy


class Synthetic(LEAF):

    def __init__(self, root):
        super(Synthetic, self).__init__(
            root,
            transform=ToNumpy(np.float32),
            target_transform=ToNumpy(np.int64)
        )
