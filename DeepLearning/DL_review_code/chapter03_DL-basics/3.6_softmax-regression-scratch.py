"""
3.6 softmax回归的从零开始实现
"""
import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

"""
3.6.1 获取和读取数据
"""
batch_size = 256
train_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, transform=torchvision.transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

"""
3.6.2 初始化模型参数
"""
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

"""
3.6.3 实现softmax运算
"""
X = torch.tensor([[1, 2, 3],[4, 5, 6]])
print(X.sum(dim=0, keepdim=True))   #对同一列元素求和，并在结果中保留行和列的维度
print(X.sum(dim=1, keepdim=True))   #对同一行元素求和，并在结果中保留行和列的维度

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))

"""
3.6.4 定义模型
"""
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)), W) + b)

"""
3.6.5 定义损失函数
"""
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0,2])
# y_hat = torch.gather(y_hat, 1, y.view(-1, 1))
print(y_hat)

def cross_entropy(y_hat, y):
    return -1 * torch.log(y_hat.gather(dim=1, index=y.view(-1, 1))).sum()

""" 
3.6.6 计算分类准确率
"""
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item() #(y_hat.argmax(dim=1) == y) =tensor([False,  True])

# print(accuracy(y_hat, y))
# print((y_hat.argmax(dim=1) == y))

def evaluate_accuracy(data_iter, net):
    '''评价模型net在数据集data_iter上的准确率'''
    acc_sum, n = 0, 0
    for X, y in data_iter:
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    return acc_sum / n

# print(evaluate_accuracy(test_iter, net))

"""
3.6.7 训练模型
"""
num_epochs, lr = 5, 0.01
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch, train_l_sum/n, train_acc_sum/n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W,b], lr)

"""
3.6.7 预测
"""
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
print(titles)