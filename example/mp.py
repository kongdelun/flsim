import torch
from torch.nn.parallel import DistributedDataParallel

from benchmark.mnist.model import MLP

print(torch.cuda.memory.list_gpu_processes())
