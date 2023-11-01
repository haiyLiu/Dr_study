"""
3.8 多层感知机 multilayer perceptron
"""

"""
3.8.2 激活函数
"""

"""
3.8.2.1 Relu函数
"""
import torch
import numpy as np
import matplotlib  
matplotlib.use('TkAgg')  # 可以更改为其他GUI后端，例如'Qt5Agg', 'GTK3Agg'等
import matplotlib.pylab as plt
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())  #在将tensor转化为numpy时，如果需要转换的tensor在计算图（requires_grad=True）中，那么这时只能先进行detach操作才能转换为numpy
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'ReLU')

y.sum().backward()
xyplot(x, y, 'grad of relu')

"""
3.8.2.2 Sigmoid函数
"""
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x, y, 'grad of sigmoid')

"""
3.8.2.3 Tanh函数
"""
y = x.tanh()
xyplot(x, y, 'tanh')

x.grad.zero_()
y.sum().backward()
xyplot(x, y, 'grad of tanh')
