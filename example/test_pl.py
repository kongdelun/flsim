import os
import torch
import torchmetrics
from pytorch_lightning import Trainer
from torch import nn, optim, linalg
import torch.nn.functional as F
from torch.nn import Module
from torchinfo import summary
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from benchmark.mnist.model import MLP


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


root = 'D:/project/python/dataset/mnist/raw'

dataset = MNIST(root, download=False, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])
net = LitAutoEncoder()
trainer = pl.Trainer(accelerator='cuda')
trainer.fit(net, train, val)
linalg.