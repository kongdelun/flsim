from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, ToTensor, Normalize

from trainer.final.v3.core import SGDActor
from utils.data.dataset import BasicFederatedDataset, get_target
from utils.data.partition import BasicPartitioner, Part
from utils.profiler import Timer

ds = CIFAR10(
    'D:/project/python/dataset/cifar10/raw',
    transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    train=False
)
dp = BasicPartitioner(get_target(ds), Part.IID_BALANCE, 100)
fd = BasicFederatedDataset(ds, dp)
net = mobilenet_v2()
s = net.state_dict()


def dispatch(a, v):
    return a.fit.remote(*v)


pool = ActorPool([SGDActor.remote(net, CrossEntropyLoss()) for _ in range(5)])
with Timer('FIT'):
    for r in pool.map(dispatch, [(s, fd[i], 32, 5) for i in fd]):
        print(r[1:])
