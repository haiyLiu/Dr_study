"""
3.12 权重衰减------解决过拟合的常用方法
"""

"""
3.12.2 高维线性回归实验
"""
import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)    #如果你需要一个正态分布的序列,需要使用np.random.normal
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

"""
3.12.3 从零实现权重衰减
"""

"""
3.12.3.1初始化模型参数
"""
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w,b]

"""
3.12.3.2定义L2范数惩罚项
"""
def l2_penalty(w):
    return (w**2).sum() / 2

"""
3.12.3.3 定义训练和测试
"""
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in data_iter:
            y_hat = net(X, w, b)
            l = loss(y_hat, y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w,b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    print('L2 norm of w:', w.norm().item())

fit_and_plot(lambd=3)

"""
3.12.4 简洁实现
"""
def fit_and_plot_pytorch(wd):
    net = nn.Linear(num_inputs, 1)  #net(inputs, outputs)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD([net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减 权重衰减可以通过优化器中的weight_decay超参数来指定。
    optimizer_b = torch.optim.SGD([net.bias], lr=lr)    # 不对偏差参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X,y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()

            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        train_ls.append(loss(net(test_features), test_labels).mean().item())
    print('L2 norm of w:', net.weight.data.norm().item())   #求w的L2范数

fit_and_plot_pytorch(wd=3)