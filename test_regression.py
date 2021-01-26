import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import regression
import importlib as imp
imp.reload(regression)
from regression import regression

# fake data
x = np.linspace(-2, 2, 80)
y = np.exp(x/2.0)*np.power(x, 3) + 2.0*np.random.randn(len(x))
x = np.reshape(x, (x.shape[0], 1))
y = np.reshape(y, (y.shape[0], 1))

# Visualization
plt.figure(figsize=(20, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+50+310")
plt.suptitle('Regression', fontsize=20)

for n in range(1, 7):
    plt.subplot(2,3,n)
    X = np.ones((x.shape[0], 1))
    for i in range(n): X = np.hstack((np.power(x, i+1), X))
    w = np.random.randn(X.shape[1], 1)
    mat = regression(y, X, w=w, n=n, lr=0.6, lam=10, stop_condition=1e-3)
    w, mu, std = mat.train()

print("Bingo")
plt.show()
