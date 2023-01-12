from torch import nn

from utils.tool import locate


def build_model(name: str, args: dict):
    args = dict(args) if args else dict()
    return locate([
        f'benchmark.model',
    ], name, args)


class CNN21(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN21, self).__init__()
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
        self.output = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2_ = x2.view(x2.size(0), -1)
        return self.output(x2_)


class CNN32(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN32, self).__init__()
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
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MnistMLP(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.input = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)

    def forward(self, x0):
        x0 = x0.view(-1, 28 * 28)
        x1 = self.input(x0)
        x2 = self.relu(x1)
        x3 = self.output(x2)
        return x3


class SyntheticMLP(nn.Module):

    def __init__(self, input_dim=60, output_dim=10, hidden_dim=128):
        super(SyntheticMLP, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self._net(x)


class Sent140MLP(nn.Module):

    def __init__(self, max_len=35, dim=25):
        super(Sent140MLP, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self.input = nn.Sequential(
            nn.Linear(max_len * dim, int(max_len * dim / 25)),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(int(max_len * dim / 25), max_len),
            nn.ReLU()
        )
        self.output = nn.Linear(max_len, 3)

    def forward(self, x):
        x = x.view(-1, self.max_len * self.dim)
        x = self.input(x)
        x = self.hidden(x)
        return self.output(x)


class RNN(nn.Module):

    """Creates an RNN model using LSTM layers for Shakespeare language models (next character prediction task).
    Args:
        vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
            Defaults to 80.
        embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding,
            Defaults to 8.
        hidden_size (int, optional): the size of hidden layer. Defaults to 256.
    Returns:
        A `torch.nn.Module`.
    Examples:
        RNN_Shakespeare(
          (embeddings): Embedding(80, 8, padding_idx=0)
          (lstm): LSTM(8, 256, num_layers=2, batch_first=True)
          (fc): Linear(in_features=256, out_features=90, bias=True)
        ), total 819920 parameters
    """

    def __init__(self, vocab_size=80, embedding_dim=8, hidden_size=256):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output

    net = nn.Sequential(
        nn.Linear(60, 90),
        nn.ReLU(),
        nn.Linear(90, 10)
    )


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
