import torch
import numpy as np

""" 
2.2.1 创建tensor
"""


# x = torch.empty(5,3)    #占用内存，随机生成

# x = torch.rand(5,3)

# x = torch.zeros(5,3,dtype=torch.long)

# x = torch.tensor([5.5,3])

# x = x.new_ones(5,3,dtype=torch.float64)

# x = torch.randn_like(x, dtype=torch.float)

# print(x)
# print(x.shape)
# print(x.size()) 


""" 
2.2.2 操作
"""
### 算术运算
# x = torch.zeros(5,3)
# y = torch.rand(5,3)
# print(x + y)
# print(torch.add(x,y))

# result = torch.empty(5,3)
# torch.add(x,y,out=result)
# print(result)

# y.add_(x)
# print(y)

### 索引
# x = torch.zeros(5,3)
# y = x[0, :] #索引出来的结果与原数据共享内存，即修改一个，另一个也会跟着修改
# y += 1
# print(y)
# print(x[0, :])  #原tensor也被修改

### 改变形状
# x = torch.zeros(5, 3)
# y = x.view(15)  #y.shape=[15]
# z = x.view(-1, 3)   #注意：view()返回的新Tensor与原Tensor虽然有不同的size,但是是共享data的
# print(x.shape, y.shape, z.shape)

# x += 1
# print(x)
# print(y)

# x_cp = x.clone().view(15)   #clone()返回一个真正新的副本，使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor
# x -= 1
# print(x)
# print(x_cp)

# x = torch.randn(1)
# print(x)
# print(x.item()) #item()将Tensor转换成一个number

"""
2.2.3 广播机制
"""
# x = torch.arange(1,3).view(1,2)
# print(x)
# y = torch.arange(1,4).view(3,1)
# print(y)
# print(x + y)    #当对两个形状不同的tensor做运算时，会触发广播机制：先适当复制元素使这两个tensor形状相同后再按元素运算

""" 
2.2.4 运算的内存开销
"""
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y = y + x   #会开辟新内存
# print(id(y) == id_before)   #False

# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y[:] = y + x    #索引操作是不会开辟新内存的
# print(id(y) == id_before)   #True

# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# torch.add(x, y, out=y)
# print(id(y) == id_before)   #True
#注：虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。

"""
2.2.5 Tensor和Numpy相互转换
"""
# tensor转numpy
# a = torch.ones(5)
# b = a.numpy()   #numpy()和from_numpy()共享相同的内存
# print(a, b)

# a += 1
# print(a, b)

# # numpy转tensor
# a = np.ones(5)
# b = torch.from_numpy(a) #共享内存
# print(a,b)
# a += 1
# print(a, b)


# c = torch.tensor(a) #不共享内存
# a += 1
# print(a, c)


"""
2.2.3 Tensor on GPU
"""
x = torch.zeros(5,3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)   #ones_like()表示创建形状与x相同的y
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))