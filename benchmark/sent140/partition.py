from utils.data.partition import BasicPartitioner


class SENT140Partitioner(BasicPartitioner):
    FEATURE_NUM = 35*25
    CLASS_NUM = 2
