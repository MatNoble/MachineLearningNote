#==================================================
#==>      Title: linear regression
#==>   SubTitle: PyTorch VS Numpy
#==>     Author: Zhang zhen                   
#==>      Email: hustmatnoble.gmail.com
#==>     GitHub: https://github.com/MatNoble
#==>       Date: 1/28/2021
#==================================================

from linear_regression_pytorch import *
from linear_regression_numpy import *

num_inputs = 2      # w1, w2
num_examples = 1000 # 样本容量
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, 
                       dtype=torch.float32) # 特征(1000, 2)
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), 
                       dtype=torch.float32) # 标签(1000, 1)

# Visualization
# plt.figure()
# plt.suptitle('Data Visualization', fontsize=20)
# set_figsize((15, 8))
# plt.subplot(1,2,1)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.scatter(features[:,0].numpy(), labels.numpy(), 20)
# plt.xlabel('$x_1$', fontsize=16)
# plt.ylabel('$y$', fontsize=16)
# plt.subplot(1,2,2)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.scatter(features[:,1].numpy(), labels.numpy(), 20)
# plt.xlabel('$x_2$', fontsize=16)
# plt.ylabel('$y$', fontsize=16)
# plt.savefig('images/visual.svg',  dpi=600, bbox_inches='tight')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       PyTorch
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 初始化参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),
                 dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

# 训练数据
lr = 0.1
num_epochs = 150
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    X, y = features, labels
    l = loss(net(X, w, b), y).sum()
    l.backward()                   
    sgd([w, b], lr, batch_size=1000)
    w.grad.data.zero_()
    b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)

print('\nloss: %f' % train_l.mean().item())
print('true_theta =', true_w + [true_b])
print('pytorch_theta =', np.hstack((w.detach().numpy().T[0], b.detach().numpy())))
print('++++++++++++++++++++++++++++')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       手撕线性回归
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# tensor --> numpy
features_numpy = np.reshape(features.numpy(), (num_examples, num_inputs))
labels_numpy   = np.reshape(labels.numpy(), (num_examples, 1))
# 最后增加一列 1
features_numpy = np.hstack((features_numpy, np.ones((num_examples, 1))))

w = np.random.normal(0, 0.01, (num_inputs, 1))
w = np.vstack((w, [0]))
mat = linear_regression(labels_numpy, features_numpy, w=w,
                        lr=0.1, lam=0, MAX_Iter=150, stop_condition=1e-8)
loss, w = mat.train()
# out put
print('loss: %f' % loss[-1])
print('numpy_theta=', w.T[0])

# plt.figure()
# mat.plot_cost(loss, range(1, 150+1))
# plt.savefig('images/cost.svg',  dpi=600, bbox_inches='tight')

plt.show()
