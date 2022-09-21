import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class FedProxLoss(CrossEntropyLoss):

    def __init__(self, u=0.01) -> None:
        super(FedProxLoss, self).__init__()
        self.u = u

    def forward(
            self, input: Tensor, target: Tensor,
            global_vector: Tensor = None,
            local_vector: Tensor = None
    ):
        loss = super(FedProxLoss, self).forward(input, target)
        if global_vector is None or local_vector is None:
            return loss
        l1 = .5 * self.u * torch.sum(torch.pow(local_vector - global_vector, 2))
        return loss + l1


class FedDynLoss(CrossEntropyLoss):

    def __init__(self, alpha=0.01) -> None:
        super(FedDynLoss, self).__init__()
        self.alpha = alpha

    def forward(
            self, input: Tensor, target: Tensor,
            global_vector: Tensor = None,
            local_vector: Tensor = None,
            local_grad_vector: Tensor = None
    ):
        loss = super(FedDynLoss, self).forward(input, target)
        if global_vector is None or local_vector is None or local_grad_vector is None:
            return loss
        l1 = torch.dot(local_grad_vector, local_vector)
        l2 = .5 * self.alpha * torch.sum(torch.pow(local_vector - global_vector, 2))
        return loss - l1 + l2
