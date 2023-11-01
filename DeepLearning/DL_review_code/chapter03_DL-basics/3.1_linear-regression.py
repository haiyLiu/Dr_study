import torch
from time import time

""" 
3.1.1 
"""
# a = torch.ones(1000)
# b = torch.ones(1000)

# start = time()
# c = torch.zeros(1000)
# for i in range(len(a)):
#     c[i] = a[i] + b[i]  #向量相加的一种方法，是将两个向量按元素逐一做标量加法
# print(time()-start)

# start = time()
# c = a + b   #向量相加的另一种方法，是将两个向量做矢量加法------更省时
# print(time()-start)


a = torch.ones(3)
b = 10
print(a + b)
