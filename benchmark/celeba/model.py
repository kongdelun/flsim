from torch import nn


class CNN(nn.Module):

    def __init__(self, class_num=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32*16*16 -> 32*8*8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Linear(32, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
