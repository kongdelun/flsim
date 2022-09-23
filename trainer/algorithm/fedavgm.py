from trainer.core.proto import FedAvg
from utils.nn.functional import linear_sum, zero_like, add


class FedAvgM(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedAvgM, self)._parse_kwargs(**kwargs)
        if avgm := kwargs['avgm']:
            self.beta = avgm.get('beta', 0.5)

    def _init(self):
        super(FedAvgM, self)._init()
        self._mom = zero_like(self._model.state_dict())

    def _aggregate(self, cids):
        grad = self._aggregator.compute()
        self._mom = linear_sum([self._mom, grad], [self.beta, 1.])
        self._model.load_state_dict(
            add(self._model.state_dict(), self._mom)
        )
        self._aggregator.reset()
