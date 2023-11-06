"""
4.6 GPU计算
"""


"""
4.6.1 计算设备
    -pytorch可以指定用来存储和计算的设备
    -默认情况下，pytorch会将数据创建在内存中，利用CPU来计算
"""
import torch
from torch import nn

print(torch.cuda.is_available())
print(torch.cuda.current_device())  # 查看当前GPU索引号，从0开始
print(torch.cuda.get_device_name(0))    # 根据索引号查看GPU名字

"""
4.6.2 Tensor的GPU计算
"""
x = torch.tensor([1, 2, 3])
print(x)
print(x.device)
x = x.cuda()    # .cuda()可以将CPU上的Tensor转换（复制）到GPU上，如果有多块GPU，我们用.cuda(i)来表示第i块GPU及相应的显存（i从0开始），cuda(0)和cuda()等价
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
# or
x =  torch.tensor([1, 2, 3]).to(device)

y = x ** 2  # 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
print(y)

# **注意：存储在不同位置的数据是不可以直接进行计算的。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
# z = y + x.cpu()

"""
4.6.3 模型的GPU计算
    -需要保证模型输入的Tensor和模型都在同一设备上
"""
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
net.cuda()
print(list(net.parameters())[0].device)

x = torch.rand(2, 3).cuda()
print(net(x))
