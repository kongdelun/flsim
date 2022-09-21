import time

from tqdm import tqdm

from utils.result import print_text


class Timer:

    def __init__(self, name: str):
        self.name = name
        self.begin_time = None

    def __enter__(self):
        self.begin_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print_text("{} cost {:.3f}s".format(self.name, time.time() - self.begin_time), 5)
        del self


class Recorder:

    def __init__(self, bar: tqdm = None):
        self.__bar = bar
        self.__values = []

    def __enter__(self):
        self.__values.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __update_bar(self, step=1):
        if self.__bar:
            self.__bar.update(step)

    def append(self, value):
        self.__values.append(value)
        self.__update_bar()

    def clear(self):
        self.__values.clear()

    def values(self):
        if len(self.__values) < 0:
            return None
        return self.__values
