"""
3.13 丢弃法
    -只在训练模型时使用.
"""

"""
3.13.2 从零开始实现
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../')
import d2lzh_pytorch as d2l

def dropout(X, drop_prob):
    '''以drop_prob的概率丢弃X中的元素'''
    X = X.float()
    assert 0 <= drop_prob <= 1  #如果满足条件0 <= drop_prob <= 1，程序正常往下运行
    keep_prob = 1 - drop_prob

    if keep_prob == 0:  #这种情况下把全部元素都丢弃
        return torch.zeros_like(X)
    
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob

X = torch.arange(16).view(2, 8)
# print(dropout(X, 0.5))

"""
3.13.2.1 定义模型参数
"""
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)

W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)

W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

"""
3.13.2.2 定义模型
"""
drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training: #只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)    # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)    # 在第二层全连接后添加丢弃层
    return (torch.matmul(H2, W3) + b3)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X,y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            if ('is training' in net.__code__.co_varnames):# 如果有is_training这个参数
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        return acc_sum / n
    
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr):
    optimizer = torch.optim.SGD(net.paramters(), lr)
    for _ in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()



"""
3.13.2.3 定义训练和测试
"""
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
mnist_train = torchvision.datasets.FashionMNIST(root='/home/lhy/Dr_study/DeepLearning/DL_review_code/chapter03_DL-basics/Datasets/', train=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/home/lhy/Dr_study/DeepLearning/DL_review_code/chapter03_DL-basics/Datasets/', train=False, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

"""
3.13.3 简洁实现
"""
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)