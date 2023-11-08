"""
5.3 多输入通道和多输出通道
"""

"""
5.3.1 多输入通道
"""
import torch
from torch import nn
import sys
sys.path.append("./DL_review_code")
import d2lzh_pytorch as d2l

def corr2d_multi_in(X, K):
    # 先计算出X和K的第0维（通道维数）,然后再和其他维数相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
])
K = torch.tensor([
    [[0, 1], [2, 3]],
    [[1, 2], [3, 4]]
])
# print(corr2d_multi_in(X, K))

"""
5.3.2 多输出通道
"""
def corr2d_multi_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K+1, K+2])
# print(K, K.shape)
# print(corr2d_multi_out(X, K))

# 假设是时间步T1
# T1 = torch.tensor([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]])
# # 假设是时间步T2
# T2 = torch.tensor([[10, 20, 30],
#                 [40, 50, 60],
#                 [70, 80, 90]])

# print(torch.stack([T1, T2], dim=2))
# print(torch.stack([T1, T2], dim=2).shape)

"""
5.3.3  1 x 1卷积层
"""
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h*w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
print(corr2d_multi_in_out_1x1(X, K))
print(corr2d_multi_out(X, K))