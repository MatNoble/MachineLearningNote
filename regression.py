import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class regression:
    
    def MSELoss(self, y, y_hat):
        return (y-y_hat).T*(y-y_hat)/2.0
    
    def Grad(self, x, y, X, y_hat):
        return (y_hat-y).T * X
    
    def stochasticGrad(self, x, y, X, y_hat):
        i = np.random.randint(0, len(x)-1)
        return (y_hat[i]-y[i]) * X[i,:]
        
    def train(self, x, y, n=1, w=[[0, 0, 0]], lr=0.2, Iter=300):
        # y_hat = X * w
        X = np.mat(np.ones(len(x))).T
        for i in range(n):
            temp = np.mat(np.power(x, i+1)).T
            X = np.hstack((temp, X))

        # vector to matrix
        x, y = np.mat(x), np.mat(y)
        x, y = x.T, y.T
        
        y_hat = X * np.mat(w).T
        plt.subplot(2,2,n)
        plt.ion()
        adaGrad = np.zeros(n+1)
        for t in range(Iter):
            # ++++++++++
            # Adagrad
            # ++++++++++
            grad = self.Grad(x, y, X, y_hat)
            temp = np.vstack((adaGrad, grad))
            adaGrad = LA.norm(temp, axis=0)
            
            # ++++++++++
            # updata w
            # ++++++++++
            wn = w
            w = wn - lr*grad/adaGrad
            # w = wn - lr*grad
            
            # ++++++++++++
            # compute loss
            # ++++++++++++
            y_hat = X * np.mat(w).T
            loss = self.MSELoss(y, y_hat)
            
            # +++++++++++++
            # Visualization
            # +++++++++++++
            if t % 5 == 0:
                # plot and show learning process
                plt.cla()
                plt.scatter(np.array(x), np.array(y))
                plt.plot(np.array(x), np.array(y_hat), 'r-', lw=4)
                plt.text(0.4, min(y), 'Loss=%.4f' %loss, fontdict={'size': 16, 'color': 'red'})
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                # plt.xlabel('x', fontsize=20)
                # plt.ylabel('y', fontsize=20)
                plt.grid(True)
                plt.pause(0.1)

            # ++++++++++
            # end condition
            # ++++++++++
            if LA.norm(w-wn) < 1E-3:
                print('\n iteration number: ', t+1)
                break
        print('w = \n', np.array(w.T))
        print('++++++++++++++++++++++++++++')
        plt.ioff()

# =====================================
# Main
# =====================================
mat = regression()
#=========================
# fake data
#=========================
x = np.linspace(-1, 1, 100)
y = np.power(x, 3) + 0.2*np.random.randn(len(x))

#=========================
# Visualization
#=========================
plt.figure(figsize=(15, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+50+310")
plt.suptitle('Regression', fontsize=20)

for n in range(1, 5):
    # w = np.random.randn(n+1)
    w = np.zeros(n+1)
    mat.train(x, y, n=n, w=w, lr=0.4)

plt.show()
