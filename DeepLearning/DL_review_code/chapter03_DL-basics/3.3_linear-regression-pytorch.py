import torch
import numpy as np
from torch import nn

"""
3.3 线性回归的简洁实现
    - torch.utils.data模块提供了有关数据处理的工具
    - torch.nn模块定义了大量的神经网络层
    - torch.nn.init模块定义了各种初始化方法
    - torch.optim模块提供了很多常用的优化算法
"""

"""
3.3.1  生成数据集
"""

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

"""
3.3.2 读取数据
"""
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)  #TensorDataset相当于对features, labels两个tensor逐项进行zip操作
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  #取数据，与TensorDataset结合使用
# for X,y in data_iter:
#     print(X, y)
#     break

"""
3.3.3 定义模型
    注意：torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。
"""
class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_features, 1) #nn.Linear(in_features, out_features)
    
    def forward(self, x):
        '''定义前向传播算法'''
        y = self.linear(x)
        return y

# 3种定义网络模型的方法
net = LinearNet(num_inputs)
print(net)

## Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此外还可以传入其他层
# )

# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))

# from collections import OrderedDict
# net = nn.Sequential(
#     OrderedDict([
#         ('linear', nn.Linear(num_inputs, 1))
#         # .......
#     ])
# )

for param in net.linear.parameters():
    print(param)

"""
3.3.4 初始化模型参数
"""
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  #也可直接修改bias的data: net[0].bias.data.fill_(0)

"""
3.3.5 定义损失函数
"""
loss = nn.MSELoss()

"""
3.3.6 定义参数优化算法
"""
import torch.optim as optim
optimizer = optim.SGD(net.linear.parameters(), lr=0.03)

# # 为不同子网络设置不同的学习率
# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},   # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
# ])

"""
有时候我们不想让学习率固定成一个常数，那该如何调整？
    1. 修改optimizer.param_groups中对应的学习率
    2. 新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer
"""
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1
print(optimizer)


"""
3.3.6 训练模型
"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() #梯度清零，等价于net.zero_grad()
        l.backward()    #对loss反向传播
        optimizer.step()    #对参数进行更新
    print('epoch %d, loss: %f' % (epoch, l.item()))

