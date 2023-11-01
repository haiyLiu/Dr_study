"""
3.9 多层感知机的从零开始实现
"""

import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

"""
3.9.1 获取和读取数据
"""
batch_size = 256
train_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, transform=torchvision.transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

"""
3.9.2 定义模型参数
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256
Wh = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
bh = torch.zeros(num_hiddens, dtype=torch.float)
Wo = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
bo = torch.zeros(num_outputs, dtype=torch.float)

params = [Wh, bh, Wo, bo]
for param in params:
    param.requires_grad_(requires_grad=True)

"""
3.9.3 定义激活函数
"""
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

"""
3.9.4 定义模型
"""
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, Wh) + bh)
    return torch.matmul(H, Wo) + bo

"""
3.9.5 定义损失函数
"""
loss = torch.nn.CrossEntropyLoss()

"""
3.9.6 训练模型
"""
num_epochs, lr = 5, 0.01
optimizer = torch.optim.SGD(params, lr)
for epoch in range(num_epochs):
    train_l_sum = 0
    train_acc_sum, n = 0, 0
    test_acc = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()    #开始优化参数
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch, train_l_sum/n, train_acc_sum/n, test_acc))
