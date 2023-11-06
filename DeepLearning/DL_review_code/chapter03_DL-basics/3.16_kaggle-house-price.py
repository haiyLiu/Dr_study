import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import d2lzh_pytorch as d2lzh

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('Datasets/house-prices/train.csv')
test_data = pd.read_csv('Datasets/house-prices/test.csv')
print(train_data.shape) #(1460, 81)
print(type(train_data))
print(test_data.shape)  #(1459, 80)

# print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))   #默认axis=0拼接方式是上下堆叠
# print(all_features.iloc[0:4, :])

"""
3.16.3 预处理数据
    -对连续数值的特征做标准化.
    -对于缺失的特征值，我们将其替换成该特征的均值.
"""
# all_features.dtypes 返回每列数据的类型
# all_features.dtypes != 'object' 返回bool值
# all_features.dtypes[all_features.dtypes != 'object'] 把数据类型不为object的各列取出来
# all_features.dtypes[all_features.dtypes != 'object'].index 取出各列的列名
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)   #对每列应用apply函数
## 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
all_features = pd.get_dummies(all_features, dummy_na=True) #利用pandas实现one hot encode的方式, # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
print(all_features.shape)   #(2919, 331)
# print(all_features.iloc[0:4, :])

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float) #DataFrame.values返回给定DataFrame的Numpy表示形式
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1,1)

"""
3.16.4 训练模型
"""
loss = nn.MSELoss()
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
        return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

"""
3.16.5 K折交叉验证
"""
def get_k_fold(k, i, X, y):
    '''返回第i折交叉验证时所需要的训练和验证数据----第i折作为验证数据,其余k-1折作为训练数据'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)   #pandas是pd.concat
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):  #共训练 k * num_epochs 次
        data = get_k_fold(k, i, X_train,y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

def train_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy() # 返回一个新的tensor，requires_grad为false
    print(preds.reshape(-1,1)[0])  #tensor是用view,numpy用reshape
    test_data['SalePrice'] = pd.Series(preds.reshape(-1,1)[:,0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('Datasets/house-prices/submission.csv', index=False)

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
train_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)









