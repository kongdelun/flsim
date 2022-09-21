import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 卷积层
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))

        # Dropout层
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)

        # 全连接层
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        """前向传播"""

        # [b, 1, 28, 28] => [b, 32, 26, 26]
        out = self.conv1(x)
        out = F.relu(out)

        # [b, 32, 26, 26] => [b, 64, 24, 24]
        out = self.conv2(out)
        out = F.relu(out)

        # [b, 64, 24, 24] => [b, 64, 12, 12]
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)

        # [b, 64, 12, 12] => [b, 64 * 12 * 12] => [b, 9216]
        out = torch.flatten(out, 1)

        # [b, 9216] => [b, 128]
        out = self.fc1(out)
        out = F.relu(out)

        # [b, 128] => [b, 10]
        out = self.dropout2(out)
        out = self.fc2(out)

        output = F.log_softmax(out, dim=1)

        return output


# 定义超参数
batch_size = 64  # 一次训练的样本数目
learning_rate = 0.0001  # 学习率
iteration_num = 5  # 迭代次数
network = Model()  # 实例化网络
print(network)  # 调试输出网络结构
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  # 优化器

# GPU 加速
use_cuda = torch.cuda.is_available()
print("是否使用 GPU 加速:", use_cuda)


def get_data():
    """获取数据"""

    # 获取测试集
    train = torchvision.datasets.MNIST(root="D:/project/python/dataset/mnist/raw", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="D:/project/python/dataset/mnist/raw", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    # 返回分割好的训练集和测试集
    return train_loader, test_loader


def train(model, epoch, train_loader):
    """训练"""

    # 训练模式
    model.train()

    # 迭代
    for step, (x, y) in enumerate(train_loader):
        # 加速
        if use_cuda:
            model = model.cuda()
            x, y = x.cuda(), y.cuda()

        # 梯度清零
        optimizer.zero_grad()

        output = model(x)

        # 计算损失
        loss = F.nll_loss(output, y)

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 打印损失
        if step % 50 == 0:
            print('Loss: {}'.format(loss))


def test(model, test_loader):
    """测试"""

    # 测试模式
    model.eval()

    # 存放正确个数
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:

            # 加速
            if use_cuda:
                model = model.cuda()
                x, y = x.cuda(), y.cuda()

            # 获取结果
            output = model(x)

            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)

            # 计算准确个数
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算准确率
    accuracy = correct / len(test_loader.dataset) * 100

    # 输出准确
    print("Test Accuracy: {}%".format(accuracy))


def main():
    # 获取数据
    train_loader, test_loader = get_data()

    # 迭代
    for epoch in range(iteration_num):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, train_loader)
        test(network, test_loader)


if __name__ == "__main__":
    main()
