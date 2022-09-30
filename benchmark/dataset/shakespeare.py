from benchmark.dataset.leaf import LEAF, ToVector


class Shakespeare(LEAF):

    def __init__(self, root):
        super(Shakespeare, self).__init__(
            root,
            transform=ToVector(),
            target_transform=ToVector(False)
        )
