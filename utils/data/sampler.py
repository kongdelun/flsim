import random
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    """Samples elements from a given list of indices, always in the same order once initialized.

    It is a :class:`Sampler` used in :class:`Dataloader`, that each partition will be fixed once initialized.

    Args:
        indices (list[int]): Indices in the whole set selected for subset
        shuffle (bool): shuffle the indices or not.
    """

    def __init__(self, indices: list[int], shuffle=False):
        super(SubsetSampler, self).__init__(None)
        self._indices = indices
        if shuffle:
            random.shuffle(self._indices)

    def __iter__(self):
        for ind in self._indices:
            yield ind

    def __len__(self):
        return len(self._indices)
