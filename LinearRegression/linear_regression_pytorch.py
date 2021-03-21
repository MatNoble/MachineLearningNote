#==================================================
#==>      Title: linear regression based on PyTorch
#==>     Author: MatNoble                  
#==>      Email: hustmatnoble.gmail.com
#==>     GitHub: https://github.com/MatNoble
#==>       Blog: https://matnoble.me
#==>       Date: 1/28/2021
#==================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def set_figsize(figsize=(8.5, 7.5)):
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 /2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
