import warnings
from datetime import datetime
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from benchmark.fds import synthetic
from benchmark.model import SyntheticMLP
from env import TB_OUTPUT
from trainer.core.actor import BasicActor
from trainer.core.aggregator import BasicAggregator
from trainer.core.trainer import FLTrainer
from utils.cache import DiskCache
from utils.nn.functional import add, flatten
from utils.nn.stats import to_numpy
from utils.select import random_select
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class Representation(FLTrainer):

    def _init(self):
        super(Representation, self)._init()
        self._writer = SummaryWriter(
            f'{TB_OUTPUT}/{self.name}'
        )
        self._cache = DiskCache(
            self.cache_size,
            f'{self._writer.log_dir}/run/{datetime.today().strftime("%H-%M-%S")}'
        )
        self._pool = ActorPool([
            BasicActor.remote(self._model, CrossEntropyLoss())
            for _ in range(self.actor_num)
        ])
        self._aggregator = BasicAggregator()

    def _state(self, cid):
        return self._model.state_dict()

    def _local_update(self, cids):
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), [
            (self._state(c), self._fds.train(c), self.local_args)
            for c in cids
        ])):
            self._cache[cid] = {
                'grad': res[0],
                'num_sample': res[1][0]
            }
            self._aggregator.update(res[0], res[1][0])

    def _select_client(self):
        return random_select(list(self._fds), s_alpha=self.sample_rate, seed=self.seed + self._k)

    def _aggregate(self, cids):
        self._model.load_state_dict(
            add(self._model.state_dict(), self._aggregator.compute())
        )
        self._aggregator.reset()
        self.plot(cids)

    def _val(self, cids):
        pass

    def _test(self):
        pass

    def plot(self, cids):
        X = to_numpy(list(map(lambda cid: flatten(self._cache[cid]['grad']), cids)))
        M = cosine_similarity(X)
        sns.clustermap(M)
        plt.show()


if __name__ == '__main__':
    fds = synthetic(dict())
    net = SyntheticMLP()
    rep = Representation(net, fds, round=100, sample_rate=1)
    rep.start()
