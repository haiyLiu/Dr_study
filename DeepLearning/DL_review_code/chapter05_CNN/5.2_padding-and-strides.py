"""
5.2 填充和步幅
    -输入形状为 nh x nw，卷积核窗口形状为 kh x kw，那么输出形状为 (nh - kh + 1) x (nw - kw + 1)
    -超参数：填充、步幅
"""
"""
5.2.1 填充 padding
    -指在输入高和宽的两侧填充元素(通常是0元素)
    -padding=[(kh-1)/2, (kw-1)/2]
    -在默认情况下，填充为0，步幅为1。
"""
import torch
from torch import nn

def comp_conv2d(conv2d, X):
    ''''计算卷积层'''
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1) #高和宽都填充1行或列
X = torch.rand(8, 8)
# print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
# print(comp_conv2d(conv2d, X).shape)

"""
5.2.2 步幅 stride
    -每次滑动的行数和列数称为步幅。
    -当高上的步幅为sh时，宽上的步幅为sw,输出形状为 inf[(nh - kh + ph + sh)/sh] x inf[(nw - kw + pw + sw)/sw] (ph和pw是padding的单侧增加的量)
    -在默认情况下，填充为0，步幅为1。
"""
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
