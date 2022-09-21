class ToTarget:
    def __init__(self, split):
        self.__split = split

    def __call__(self, x):
        if self.__split == 'letters':
            return x - 1
        else:
            return x
