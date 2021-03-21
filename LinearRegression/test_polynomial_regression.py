import pandas as pd
from polynomial_regression import *
# import polynomial_regression
# import importlib as imp
# imp.reload(polynomial_regression)
# from polynomial_regression import regression

# 多项式拟合
# fake data
x = np.linspace(-2, 2, 150)
y = np.exp(x/2.0)*np.power(x, 3) + 2.0*np.random.randn(len(x))
x = np.reshape(x, (x.shape[0], 1))
y = np.reshape(y, (y.shape[0], 1))

# Visualization
plt.figure(figsize=(18, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+50+310")
plt.suptitle('Regression', fontsize=20)

for n in range(1, 7):
    plt.subplot(2,3,n)
    X = np.ones((x.shape[0], 1))
    for i in range(n): X = np.hstack((np.power(x, i+1), X))
    w = np.random.randn(X.shape[1], 1)
    mat = regression(y, X, w=w, n=n, lr=0.4, lam=0, stop_condition=1e-3)
    w, mu, std = mat.train()

# X = np.ones((x.shape[0], 1))
# for i in range(6): X = np.hstack((np.power(x, i+1), X))
# w = np.random.randn(X.shape[1], 1)
# mat = regression(y, X, w=w, n=n, lr=0.4, lam=10, stop_condition=1e-3)
# w, mu, std = mat.train()

print("Bingo")
plt.show()

# def load_data(filename):
#     df = pd.read_csv(filename, sep=",", index_col=False)
#     data = np.array(df, dtype=float)
#     data = data[data[:, 0].argsort()]
#     return data[:, 0], data[:, -1]

# x, y = load_data("house_price_data.csv")
# x = np.reshape(x, (x.shape[0], 1))
# y = np.reshape(y, (y.shape[0], 1))
# X = np.hstack((x, np.ones((x.shape[0], 1))))

# w = np.random.randn(X.shape[1], 1)
# # w = np.zeros((X.shape[1], 1))
# mat = regression(y, X, w=w, n=1, lr=1.5, lam=0, stop_condition=1e-1)
# w, mu, std = mat.train()

# plt.show()
