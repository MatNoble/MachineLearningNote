import logistic_regression_numpy
import importlib as imp
imp.reload(logistic_regression_numpy)
from logistic_regression_numpy import *

n = 100
n_data = np.ones((n, 2))
x0 = np.random.normal(2*n_data, 0.9)
y0 = np.zeros(n)
x1 = np.random.normal(-2*n_data, 0.9)
y1 = np.ones(n)
x = np.concatenate((x0, x1), 0)
x = np.concatenate((x, np.ones((2*n, 1))), 1)
y = np.concatenate((y0, y1), 0)

theta = np.random.normal(0, 0.01, (2, 1))
theta = [[0.5],[-0.1]]
theta = np.concatenate((theta, [[0.]]), 0)
lr = 0.04
plt.figure(figsize=(8.5, 7.5))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+350+350")
mat = logisticRegression(np.reshape(y, (-1,1)), x, theta,
                         lr, lam=0, MAX_Iter=100, stop_condition=1e-2)
t, loss, w = mat.train()
print(w)

# plt.figure()
# mat.plot_cost(loss, range(1, t+1))
plt.show()
