""" 
3.5 图像分类数据集
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..') # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l


"""
3.5.1 获取数据集
    - torchvision.datasets: 一些加载数据的函数及常用的数据集接口
    - 通过train参数指定获取训练集或测试集。
    - transform=transforms.ToTensor()将所有数据转换为tensor,如果不进行转换则返回的是PIL图片。transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。
    - Fashion-MNIST是一个10类服饰分类数据集
"""
mnist_train = torchvision.datasets.FashionMNIST(root='/home/lhy/Dr_study/DeepLearning/DL_review_code/chapter03_DL-basics/Datasets/', train=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/home/lhy/Dr_study/DeepLearning/DL_review_code/chapter03_DL-basics/Datasets/', train=False, transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label) # Chanel x Height x Width

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

"""
3.5.2读取小批量数据
"""
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)

start = time.time()
for X,y in train_iter:
    continue
print('%.2f sec.' % (time.time()-start))


