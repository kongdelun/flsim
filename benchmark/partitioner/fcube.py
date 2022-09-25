import random

import numpy as np

import utils.data.functional as F
from utils.data.partition import DataPartitioner, Part


class FCUBEPartitioner(DataPartitioner):
    """FCUBE data partitioner.

    FCUBE is a synthetic dataset for research in non-IID scenario with feature imbalance. This
    dataset and its partition methods are proposed in `Federated Learning on Non-IID Data Silos: An
    Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Supported partition methods for FCUBE:

    - feature-distribution-skew:synthetic

    - IID

    For more details, please refer to Section (IV-B-b) of original paper.

    Args:
        data : Data of dataset :class:`FCUBE`.
    """

    def __contains__(self, item):
        pass

    def __iter__(self):
        pass

    # only accept partition for 4 clients
    CLASS_NUM = 2
    CLIENT_NUM = 4

    def __init__(self, data, partition, **kwargs):
        super(FCUBEPartitioner, self).__init__()
        if partition not in [Part.NONIID_SYNTHETIC, Part.IID_BALANCE]:
            raise ValueError(f"FCUBE only supports 'synthetic' and 'iid' partition, not {partition}.")
        self.partition = partition
        self.data = data
        self.seed = kwargs.get('seed', 2077)
        self.sample_num = data.shape[0] if isinstance(data, np.ndarray) else len(data)
        self.client_dict = self._split()

    def _split(self):
        if self.partition == Part.NONIID_SYNTHETIC:
            # feature-distribution-skew:synthetic
            return F.noniid_fcube_synthetic_partition(self.data)
        else:
            # IID partition
            client_sample_nums = F.balance_split(FCUBEPartitioner.CLIENT_NUM, self.sample_num)
            return F.iid_partition(client_sample_nums, self.sample_num)

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return FCUBEPartitioner.CLIENT_NUM
