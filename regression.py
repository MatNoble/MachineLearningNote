import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class regression:
    def __init__(self, y, X, w=np.zeros((3, 1)), n=1, lr=0.2, lam=0, MAX_Iter=500, stop_condition=1e-4):
        # axis
        self.x, self.y = X[:,-2], y
        # matrix
        self.X, self.mu, self.std = self.normalize(X)
        # polynomial of degree n
        self.n = n
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

    def normalize(self, data):
        mu = std = []
        for i in range(data.shape[1]-1):
            data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:, i])
            mu.append(np.mean(data[:,i]))
            std.append(np.std(data[:, i]))
        return data, mu, std
        
    def train(self):
        t, w = 1, self.w
        y_hat = self.X @ w
        plt.ion()
        while True:
            # updata w
            wn = w
            w = wn - self.lr * self.Grad(y_hat, w.copy())
            
            # compute loss
            y_hat = self.X @ w
            loss = self.MSELoss(y_hat, w)
            
            # Visualization
            if t % 5 == 0: self.visual(y_hat, loss)
           
            # end condition
            if LA.norm(w-wn) < self.stop_condition or t > self.MAX_Iter:
                print('\n iteration number: ', t)
                break
            t += 1
        # out put
        print('w = \n', w)
        print('++++++++++++++++++++++++++++')
        plt.ioff()
        return w, self.mu, self.std

    def MSELoss(self, y_hat, w):
        m = 0.5/self.y.shape[0]
        return m*((self.y-y_hat).T @ (self.y-y_hat) + self.lam * self.Regularization(w))

    def Regularization(self, w, k=1):
        if k==1: # L2
            return w.T @ w
        # elif k==2: # L1
    
    def Grad(self, y_hat, w):
        w[-1] = 0
        m = 1.0/self.y.shape[0]
        return m*(self.X.T @ (y_hat-self.y) + self.lam * w)
    
    def visual(self, y_hat, loss):
        # plot and show learning process
        plt.cla()
        plt.scatter(self.x, self.y)
        plt.plot(self.x, y_hat, 'r-', lw=4)
        plt.text(-1.5, 17.5, 'Loss=%.5f' %loss, fontdict={'size': 18, 'color': 'red'})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.pause(0.1)

if __name__ == "__main__":
    # =====================================
    # Main
    # =====================================
    # fake data
    x = np.linspace(-2, 1, 200)
    y = np.power(x, 2) + 0.2*np.random.randn(len(x))
    # y = np.exp(x/2.0)*np.power(x, 3) + 0.4*np.random.randn(len(x))
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    
    # Visualization
    plt.figure(figsize=(15, 10))
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+50+310")
    plt.suptitle('Regression', fontsize=20)
    
    for n in range(1, 5):
        plt.subplot(2,2,n)
        X = np.ones((x.shape[0], 1))
        for i in range(n):
            temp = np.power(x, i+1)
            X = np.hstack((temp, X))
        w = np.random.randn(X.shape[1], 1)
        # w = np.zeros((X.shape[1], 1))
        mat = regression(y, X, w=w, n=n, lr=0.01, lam=100)
        mat.train()
    
    print("Bingo")
    plt.show()
