"""
4.1 模型构造
    -继承Module类
    -ModuleList
    -ModuleDict
    -Sequential
"""

"""
4.1.1 继承Moudle类来构造模型
"""
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
        
    def forward(self, X):
        a = self.act(self.hidden(X))
        return self.output(a)


X = torch.rand(2, 784)
net = MLP()
# print(net)
# print(net(X))  #调用MLP继承Module类的__call__函数,使用该函数完成前向计算

"""
4.1.2 Module的子类
"""
"""
4.1.2.1 Sequential类
    -它可以接收一个子模块的<有序>字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
"""
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, moudle in args[0].items():
                self.add_module(key, moudle)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
# print(net)
# net(X)


"""
4.1.2.2 ModuleList类
    -ModuleList接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作
    -仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现
"""
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
# print(net[-1])
# print(net)

class MyModule(nn.Module):
    def __init__(self) -> None:
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])    #有10个Linear模型
    
    def forward(self, X):
        for i, l in enumerate(self.linears):
            X = self.linears[i // 2](X) + l(X)
        return X

class Module_ModuleList(nn.Module):
    def __init__(self) -> None:
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10)])

class Module_List(nn.Module):
    def __init__(self) -> None:
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10,10)]

net3 = MyModule()
net1 = Module_ModuleList()
net2 = Module_List()
# print('net1:')
# for p in net1.parameters():
#     print(type(p.data), p.size())
'''
加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
net1:
<class 'torch.Tensor'> torch.Size([10, 10])
<class 'torch.Tensor'> torch.Size([10])
'''

# print('net2:')
# for p in net2.parameters():
#     print(type(p.data), p.size())

"""
4.1.2.3 ModuleDict类
    -和ModuleList一样，ModuleDict实例仅仅是存放了一些模块的字典，并没有定义forward函数需要自己定义。
    -ModuleDict里的所有模块的参数会被自动添加到整个网络中。
    -无序的模块都需要自己定义forward函数
"""
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10)
# print(net['linear'])
# print(net.output)
# print(net)

"""
4.1.3 构造复杂的模型
"""
class FancyMLP(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        X = nn.functional.relu(torch.mm(X, self.rand_weight.data) + 1)  #rand_weight（注意它不是可训练模型参数）

        X = self.linear(X)  ## 复用全连接层。等价于两个全连接层共享参数(权重)
        while X.norm().item() > 1:
            X /= 2
        if X.norm().item() < 0.8:
            X *= 10
        return X.sum()

X = torch.rand(2, 20)
net = FancyMLP()
# print(net)
# print(net(X))

class NestMLP(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(40, 30),
            nn.ReLU()
        )
    
    def forward(self, X):
        return self.net(X)

X = torch.rand(2, 40)
net = nn.Sequential(
    NestMLP(),
    nn.Linear(30, 20),
    FancyMLP()
)
print(net)
print(net(X))



