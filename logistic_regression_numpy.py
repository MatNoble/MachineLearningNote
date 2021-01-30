import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self, y, X, theta, lr,
                 lam, MAX_Iter=500, stop_condition=1e-4):
        self.X = X
        self.y = y
        self.theta = theta
        self.lr = lr
        self.lam = lam
        self.MAX_Iter = MAX_Iter
        self.stop_condition = stop_condition

    def train(self):
        t, theta = 1, self.theta
        z = self.X @ theta
        y_hat = self.activationFunc(z)
        Loss = []
        plt.ion()
        while True:
            # update theta
            thetan = theta
            theta = thetan - self.lr * self.Grad(y_hat)
            # print(theta.shape)ones

            # compute loss
            z = self.X @ theta
            y_hat = self.activationFunc(z)
            loss = self.CrossEntropy(y_hat)
            Loss.append(loss[0][0])

            if t % 2 == 0:
                # plot and show learning process
                plt.cla()
                pred_y = np.where(y_hat.copy() > 0.5, 1., 0.)
                target_y = self.y
                plt.scatter(self.X[:, 0], self.X[:, 1],
                            c=np.reshape(pred_y, (pred_y.shape[0],)),
                            s=100, lw=0, cmap='RdYlGn')
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
                # if t == 2: plt.pause(1)
                plt.pause(0.1)
            
            # end condition
            if LA.norm(theta-thetan) < self.stop_condition or t == self.MAX_Iter:
                print('\n iteration number: ', t)
                break
            t += 1
        plt.ioff()
        return t, Loss, theta

    def CrossEntropy(self, y_hat):
        m = -1.0/self.y.shape[0]
        return m*(self.y.T @ np.log(y_hat) + (1-self.y).T @ np.log(1-y_hat))

    def activationFunc(self, z):
        return 1.0 / (1+np.exp(-z))

    def Grad(self, y_hat):
        m = 1.0/self.y.shape[0]
        return m*(self.X.T @ (y_hat-self.y))
    
    def plot_cost(self, J_all, num_epochs):
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.plot(num_epochs, J_all, 'm', linewidth = "5")
