from utils.data.partition import BasicPartitioner


class CELEBAPartitioner(BasicPartitioner):
    FEATURE_NUM = 3 * 218 * 178
    CLASS_NUM = 2
