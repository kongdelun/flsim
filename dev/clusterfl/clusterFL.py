from collections import OrderedDict

import torch
from torch import optim
from torch.nn import Module
from torch.utils.data import Dataset

from trainer.core.actor import CPUActor
from trainer.core.aggregator import Aggregator, NotCalculated
from utils.nn.functional import sub, flatten, numel


class ClusteredFLActor(CPUActor):

    def __init__(self, model: Module, loss: Module, M: int, alpha: float = 0.001, rho: float = 0.002):
        super().__init__(model, loss)
        self.alpha = alpha
        self.rho = rho
        self.M = M
        self.D = numel(self.model.state_dict())

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        opt = args.get('opt', {'lr': 0.001})
        batch_size = args.get('batch_size', 32)
        epoch = args.get('epoch', 5)
        F_i = args.get('F_i', torch.zeros(self.M))
        U = args.get('U', torch.zeros((self.M, self.D)))
        Omega = args.get('Omega', torch.zeros((self.M, self.D)))
        self.set_state(state)
        opt = optim.SGD(self.model.parameters(), **opt)
        self.model.train()
        for k in range(epoch):
            for data, target in self.dataloader(dataset, batch_size):
                opt.zero_grad()
                loss = self.loss(self.model(data), target) + self._fixed_term(F_i, U, Omega)
                loss.backward()
                opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, batch_size)

    def _fixed_term(self, F_i, U, Omega):
        W = flatten(self.get_state())
        c1 = self.alpha + .5 * self.rho * torch.sum(torch.pow(F_i, 2))
        t = c1 * torch.sum(torch.pow(W, 2))
        for j in range(self.M):
            t += F_i[j] * torch.dot(U[j] + self.rho * Omega[j], W)
        return t


class ClusteredFLAggregator(Aggregator):

    def __init__(self):
        super().__init__()
        self.alpha = None
        self._state = None
        self.F = None
        self.U = None
        self.beta = None
        self.rho = None
        self.Omega = None
        self.M = None
        self._k = -1

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        super(ClusteredFLAggregator, self).update()

    def _flash_F(self):
        pass

    def compute(self):
        try:
            return super(ClusteredFLAggregator, self).compute()
        except NotCalculated:
            # Solve for Omega_t+1
            W = flatten(self._state)
            for j in range(self.M):
                self.Omega[j] = (1.0 / (self.rho - 2 * self.beta)) * (self.rho * torch.dot(self.F[:, j], W) - self.U[j])
            # Solve for U_t+1
            for j in range(self.M):
                self.U[j] += self.rho * (self.Omega[j] - torch.dot(self.F[:, j], W))
            self._k += 1
            if self._k % 3 == 0:
                self._flash_F()
            omega = torch.dot(self.F.T, W)
            regular = self.alpha * torch.trace(torch.dot(W, W.T)) - self.beta * torch.trace(torch.dot(omega, omega.T))
            # for i in range(self.M):
            #     if update_flag[i] == 1:
            #         conver_indicator += np.abs(Loss[i] - Loss_cache[i])

# model = MLP()
#
# M = 100
# D = numel(model.state_dict())
# F_i = torch.zeros(M)
# U = torch.zeros((M, D))
# Omega = torch.zeros((M, D))
# #
# # print(Omega[3])
# alpha = 1 * (1e-3)
# beta = 5 * (1e-4)
# rho = 2 * (1e-3)
#
# print(alpha)
