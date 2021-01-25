import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class regression:
       
    def train(self, x, y, X, n=1, w=np.mat(np.zeros(3)), lr=0.2, lam=0, Iter=500):
        y_hat = X * w.T
        
        plt.ion()
        adaGrad = np.mat(np.zeros(n+1))
        for t in range(Iter):
            # Adagrad
            grad = self.Grad(lam, w.copy(), x, y, X, y_hat)
            temp = np.vstack((adaGrad, grad))
            adaGrad = LA.norm(temp, axis=0)
            
            # updata w
            wn = w
            w = wn - lr*grad/adaGrad
            # w = wn - lr*grad
            
            # compute loss
            y_hat = X * w.T
            loss = self.MSELoss(lam, w, y, y_hat)
            
            # Visualization
            if t % 5 == 0: self.visual(x, y, y_hat, loss)
           
            # end condition
            if LA.norm(w-wn) < 1E-4 or t == Iter-1:
                print('\n iteration number: ', t+1)
                break
        # out put
        print('w = \n', w.T)
        print('++++++++++++++++++++++++++++')
        plt.ioff()

    def MSELoss(self, lam, w, y, y_hat):
        return (y-y_hat).T*(y-y_hat)/2.0/len(y) + lam*self.Regularization(w)/2.0/len(y)

    def Regularization(self, w, k=1):
        if k==1: # L2
            temp = np.asarray(w)[0][:-1]
            return np.power(LA.norm(temp), 2)
        # elif k==2: # L1
    
    def Grad(self, lam, w, x, y, X, y_hat):
        w[0, -1] = 0
        return (y_hat-y).T * X/len(y) + lam*w/len(y)
    
    # def stochasticGrad(self, x, y, X, y_hat):
    #     i = np.random.randint(0, len(x)-1)
    #     return (y_hat[i]-y[i]) * X[i,:]

    def visual(self, x, y, y_hat, loss):
        # plot and show learning process
        plt.cla()
        plt.scatter(np.array(x), np.array(y))
        plt.plot(np.array(x), np.array(y_hat), 'r-', lw=4)
        plt.text(0.4, min(y), 'Loss=%.5f' %loss, fontdict={'size': 16, 'color': 'red'})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.pause(0.1)

# =====================================
# Main
# =====================================
mat = regression()
# fake data
x = np.linspace(-1, 1, 200)
y = np.exp(x*x*x)*np.power(x, 3) + 0.4*np.random.randn(len(x))

# vector to matrix
x, y = np.mat(x), np.mat(y)
x, y = x.T, y.T

# Visualization
plt.figure(figsize=(15, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+50+310")
plt.suptitle('Regression', fontsize=20)

for n in range(1, 5):
    X = np.mat(np.ones(len(x))).T
    for i in range(n):
        temp = np.mat(np.power(x, i+1))
        X = np.hstack((temp, X))
    # w = np.mat(np.random.randn(n+1))
    plt.subplot(2,2,n)
    w = np.mat(np.zeros(n+1))
    mat.train(x, y, X, n=n, w=w, lr=0.4, lam=0.1)

print("Bingo")
plt.show()
