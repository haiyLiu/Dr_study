import torch

"""
2.3 自动求梯度
    在深度学习中，我们经常需要对函数求梯度（gradient），PyTorch提供的autograd包能够根据输入和前向传播过程自动构建计算图，并执行反向传播
        1. Tensor是这个包的核心，如果将其属性.requires_grad设置为True，它将开始追踪在其上的所有操作，并且具有传递性，比如x的requires_grad=True，y由x计算得出，则y的requires_grad也为True。完成计算后，可以调用.backward()完成所有梯度的计算，Tensor的梯度将会累积到.grad属性中。
        2. 如果不想被追踪，可以调用.detach()将其从追踪记录中分离出来，不会计算梯度。可以用with torch.no_grad()将不想被追踪的代码包裹起来，在评估模型时经常使用。
        3. Function类很重要。Tensor和Function互相结合就可以构建一个有整个计算过程的有向无环图(DAG)。每个Tensor都有一个.grad_fn属性，该属性即创建Tensor的Function，就是说该Tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
"""

"""
2.3.1 Tensor
"""
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn)    #x是直接创建的，所以它没有grad_fn

# y = x + 2
# print(y)
# print(y.grad_fn)    #y有一个AddBackward0的grad_fn

# print(x.is_leaf, y.is_leaf) #True False，像x这种直接创建的称为叶子结点

# z = y * y * 3
# out = z.mean()
# print(z, out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)  #False
# a.requires_grad_(True)
# print(a.requires_grad)  #True
# b = (a * a).sum()
# print(b.grad_fn)

"""
2.2.2 梯度
    y.backward()时，
        若y是标量，则不需要为backward()传入任何参数；
        若y是张量，则需要传入一个与y同形的Tensor
"""
# out.backward()  #对运算最后的result使用backward()，就能把前面所有变量的梯度计算出来
# print(x.grad)   #tensor([[4.5000, 4.5000],
#                 # [4.5000, 4.5000]])

# out2 = x.sum()
# out2.backward()
# print(x.grad)   #tensor([[5.5000, 5.5000],
#                 # [5.5000, 5.5000]])
# grad在每次反向传播的过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播前把梯度清零。

# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)   #tensor([[1., 1.],
#                 # [1., 1.]])

# x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# y = 2 * x
# z = y.view(2, 2)
# print(z)

# v = torch.tensor([
#     [1.0, 0.1],
#     [0.01, 0.001]
# ], dtype=torch.float)
# z.backward(v)   #z不是标量，传入一个和z同形的权重向量
# print(x.grad)


# x = torch.tensor(1.0, requires_grad=True)
# y1 = x ** 2
# with torch.no_grad():
#     y2 = x ** 3

# y3 = y1 + y2
# print(x.requires_grad)
# print(y1, y1.requires_grad)
# print(y2, y2.requires_grad) #False
# print(y3, y3.requires_grad)
# y3.backward()
# print(x.grad)

x = torch.ones(1, requires_grad=True)
print(x.data)   #仍是tensor
print(x.data.requires_grad) #False

y = 2 * x
x.data *= 100   #只改变了x的值,不会记录在计算图中,不会影响梯度传播
y.backward()
print(x)
print(x.grad)
print(y)
