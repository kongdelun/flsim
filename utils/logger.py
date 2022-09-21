import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)


class SPyLogger(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.warning('init')


spy = SPyLogger()
