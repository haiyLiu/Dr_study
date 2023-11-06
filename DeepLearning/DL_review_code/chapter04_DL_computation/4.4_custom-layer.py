"""
4.4 自定义层
"""

"""
4.4.1 不含模型参数的自定义层
"""
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CenteredLayer, self).__init__(**kwargs)   #该模型没有参数,因为没有定义层
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
y_hat = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
# print(y_hat)

net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)
y = net(torch.rand(4, 8))
# print(y.mean().item())

"""
4.4.2 含模型参数的自定义层
"""
class MyDense(nn.Module):
    def __init__(self) -> None:
        super(MyDense, self).__init__()
        # self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        # self.params.append(nn.Parameter(torch.randn(4, 1)))

        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({
            'linear3': nn.Parameter(torch.randn(4, 2))
        })
    
    def forward(self, x, choice='linear1'):
        # for i in range(len(self.params)):
        #     x = torch.mm(x, self.params[i])
        # return x

        return torch.mm(x, self.params[choice])

net = MyDense()
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))