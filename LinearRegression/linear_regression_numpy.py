#==================================================
#==>      Title: linear regression based on Numpy
#==>     Author: MatNoble                  
#==>      Email: hustmatnoble.gmail.com
#==>     GitHub: https://github.com/MatNoble
#==>       Blog: https://matnoble.me
#==>       Date: 1/28/2021
#==================================================

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class linear_regression:
    def __init__(self, y, X, w=np.zeros((3, 1)),
                 lr=0.2, lam=0, MAX_Iter=500, stop_condition=1e-4):
        # axis
        self.x, self.y = X.copy(), y
        # matrix
        self.X = X
        # weights
        self.w = w
        # learning rate
        self.lr = lr
        # lambda
        self.lam = lam
        # The maximum number of iterations
        self.MAX_Iter = MAX_Iter
        # stop condition
        self.stop_condition = stop_condition
        
    def train(self):
        t, w = 1, self.w
        y_hat = self.X @ w
        Loss = []
        while True:
            # updata w
            wn = w
            w = wn - self.lr * self.Grad(y_hat, w.copy())
            
            # compute loss
            y_hat = self.X @ w
            loss = self.MSELoss(y_hat, w)
            Loss.append(loss[0][0])
            
            # end condition
            if LA.norm(w-wn) < self.stop_condition or t == self.MAX_Iter:
                print('\n iteration number: ', t)
                break
            t += 1
        return Loss, w
    
    def MSELoss(self, y_hat, w):
        m = 0.5/self.y.shape[0]
        return m*((self.y-y_hat).T @ (self.y-y_hat) \
                  + self.lam * self.Regularization(w))
    
    def Regularization(self, w, k=1):
        if k==1: # L2
            return w.T @ w
        # elif k==2: # L1
    
    def Grad(self, y_hat, w):
        w[-1] = 0
        m = 1.0/self.y.shape[0]
        return m*(self.X.T @ (y_hat-self.y) + self.lam * w)
    
    def plot_cost(self, J_all, num_epochs):
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.plot(num_epochs, J_all, 'm', linewidth = "5")
