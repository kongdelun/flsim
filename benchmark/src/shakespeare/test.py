from torch import optim, nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from benchmark.dataset.shakespeare.shakespeare import Shakespeare
from benchmark.model.nlp import RNN

shake = Shakespeare('D:/project/python/dataset/shakespeare/raw', 0.1)
print(len(shake.users))

net = RNN().cuda()

acc = Accuracy().cuda()
opt = optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss().cuda()
for _ in range(100):
    net.train()
    for data, target in DataLoader(shake.test(), batch_size=32):
        data, target = data.cuda(), target.cuda()
        opt.zero_grad()
        logit = net(data)
        loss = loss_fn(logit, target)
        acc.update(logit, target)
        loss.backward()
        opt.step()
    print(acc.compute())
