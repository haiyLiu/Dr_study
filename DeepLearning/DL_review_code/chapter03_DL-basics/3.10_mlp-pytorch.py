"""
3.9 多层感知机的简洁实现
"""
import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

"""
3.10.1 定义模型
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = torch.nn.Sequential(
    d2l.FlattenLayer(),
    torch.nn.Linear(num_inputs, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, num_outputs)
)

for params in net.parameters():
    torch.nn.init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epoches = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size)