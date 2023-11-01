"""
3.7 softmax回归的简洁实现
"""
import torch
import torchvision
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

"""
3.7.1 获取和读取数据
"""
batch_size = 256
train_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, transform=torchvision.transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

"""
3.7.2 初始化模型参数
"""
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, X): # X.shape=(batch_size, 1, 28, 28)
        X = X.view((X.shape[0], -1))
        y = self.linear(X)
        return y
net = LinearNet(num_inputs, num_outputs)

from collections import OrderedDict
# net = nn.Sequential(
#     OrderedDict([
#         # ('flatten', d2l.FlattenLayer),
#         ('linear', nn.Linear(num_inputs, num_outputs))
#     ])
# )
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

"""
3.7.3 softmax运算和交叉熵损失函数
"""
loss = nn.CrossEntropyLoss()    # 包括softmax运算和交叉熵损失计算的函数

"""
3.7.4 定义优化算法
"""
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

"""
3.7.5训练模型
"""
num_epochs = 5
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    print('epoch %d, train_loss %.4f, train acc %.4f' % (epoch, train_l_sum/n, train_acc_sum/n))




