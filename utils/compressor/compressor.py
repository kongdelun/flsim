from abc import ABC


class Compressor(ABC):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError