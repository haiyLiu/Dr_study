
"""
4.2 模型参数的访问 初始化和共享
"""
import torch
from torch import nn
from torch.nn import init


"""
4.2.1 访问模型参数
"""
net = nn.Sequential(
    nn.Linear(4, 3),    # net[0]
    nn.ReLU(),          # net[1]
    nn.Linear(3, 1)     # net[2]
)   #pytorch已经默认初始化参数
# print(net)

X = torch.rand(2, 4)
Y = net(X).sum()

# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print(name, param)
# for name, param in net[0].named_parameters():
#     print(name, param, type(param))

# class MyModel(nn.Module):
#     def __init__(self) -> None:
#         super(MyModel, self).__init__()
#         self.weight1 = nn.Parameter(torch.rand(20, 20)) #如果一个Tensor是Parameter,那么它将会自动被添加到模型的参数里
#         self.weight2 = torch.rand(20, 20)
    
#     def forward(self, X):
#         pass
# n = MyModel()
# for name, param in n.named_parameters():    #weight2不在里面
#     print(name, param.size(), type(param))

weight_0 = list(net[0].parameters())[0]
# print(weight_0.data)
# print(weight_0.grad)
# Y.backward()
# print(weight_0.grad)

"""
4.2.2 初始化模型参数
"""
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    elif 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

"""
4.2.3 自定义初始化方法
"""
def init_weight(t):
    with torch.no_grad():
        t.uniform_(-10, 10)
        t *= (t.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight(param)
        print(name, param.data)
    elif 'bias' in name:
        param.data += 1
        print(name, param.data)

"""
4.2.4 共享模型参数
"""
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(    #调用同一个Module实例,参数也是共享的
    linear,
    linear
)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1])) #在内存中，这两个线性层其实一个对象
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)   # 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的