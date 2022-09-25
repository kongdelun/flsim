from torch import nn


class FCUBE(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 9),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(9, 2)
        )

    def forward(self, x):
        return self.net(x)
