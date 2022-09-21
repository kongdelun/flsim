from torch import nn
from torchinfo import summary


class CNN(nn.Module):

    def __init__(self, out_dim=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(7 * 7 * 32, out_dim)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2_ = x2.view(x2.size(0), -1)
        return self.output(x2_)


