"""
4.5 读取和存储
"""

"""
4.5.1 读写Tensor
"""
import torch
from  torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save(
    {'x': x, 'y': y},
    'xy_dict.pt'
)
xy_dict = torch.load('xy_dict.pt')
print(xy_dict)

"""
4.5.2 读写模型
"""
"""
4.5.2.1 state_dict
    -它是一个从超参数名称 隐射到 超参数Tensor的字典对象
    -只有可学习参数的层才有state_dict中的条目,比如ReLU()函数没有state_dict值,也就是说只要直接对超参数有影响的层,就会被记录进去
    -优化器也有一个state_dict,其中包含优化器状态以及使用的超参数信息
"""
class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

"""
4.5.2.2 保存和加载模型
PyTorch中保存和加载训练模型有两种常见的方法:
    -仅保存和加载模型参数(state_dict)
    -保存和加载整个模型
"""
# 1.保存和加载模型参数（推荐＊）
X = torch.randn(2, 3)
Y = net(X)

PATH = 'net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)

# 2.保存和加载整个模型
torch.save(net, 'model.pt')
net3 = torch.load('model.pt')
print(net3)


