
"""
3.2 线性回归的从零开始实现
"""

"""
3.2.1 生成数据集
"""
import torch
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib_inline
import numpy as np
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)   # np.random.normal(正态分布的均值，正态分布的标准差，输出的维度)
# # print(features[0], labels[0])

# def use_svg_display():
#     matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     plt.rcParams['figure.figsize'] = figsize

# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

""" 
3.2.2 读取数据
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #保证样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j) #index_select()的第一个参数0表示以行为标准选择，yield是return和迭代器的结合，以便用户想要一次取一个值，然后在后续需要的时候取下面一个值

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

"""
3.2.3 初始化模型参数
    将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
"""
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

"""
3.2.4 定义模型
"""
def linreg(X, w, b):
    return torch.mm(X, w) + b   #torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。

"""
3.2.5 定义损失函数
"""
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

"""
3.2.6 定义优化算法
"""
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  #在进行值的改变时，使用.data,这样此操作不会被记录在计算图中

"""
3.2.7 训练模型
"""
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    #使用小批次的样本训练更新参数
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        ## 不要忘记将梯度清0
        w.grad.data.zero_()
        b.grad.data.zero_()
    
    train_l = loss(net(features, w, b), labels) #loss是参数更新后产生的误差
    print("Epoch %d, loss %f" % (epoch+1, train_l.mean().item())) #item()将Tensor转换成一个number

print(true_w, '\n', w)
print(true_b, '\n', b)



