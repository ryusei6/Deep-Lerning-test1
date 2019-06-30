from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn import linear_model

np.random.seed(0)

x = np.linspace(-10, 10, 50)
y_truth = 0.001 * (x **3 + x**2 + x)
y_data_plot = y_truth + np.random.normal(0, 0.05, len(x))

# plt.figure(figsize=(7, 5))
# plt.plot(x, y_truth, color='gray')
# plt.plot(x, y, '.', color='k')
# plt.show()


x2 = np.linspace(-10, 10, 200)

# 表示フォーマットを指定
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

poly = PolynomialFeatures(degree=30, include_bias=False) #30乗の多項式 [x,x^2,...,x^30]
x_poly = poly.fit_transform(x.reshape(-1,1))

model_plain = linear_model.LinearRegression(normalize=True)
model_plain.fit(x_poly, y_data_plot)
y_plain = model_plain.predict(poly.fit_transform(x2.reshape(-1,1)))
print('No Regularization Model')
print(model_plain.coef_ * 10000)

model_lasso = linear_model.LassoLars(normalize=True, alpha=0.001)
model_lasso.fit(x_poly, y_data_plot)
y_lasso = model_lasso.predict(poly.fit_transform(x2.reshape(-1,1)))
print('L1')
print(model_lasso.coef_ * 10000)

model_ridge = linear_model.Ridge(normalize=True, alpha=0.5)
model_ridge.fit(x_poly, y_data_plot)
y_ridge = model_ridge.predict(poly.fit_transform(x2.reshape(-1,1)))
print('L2')
print(model_ridge.coef_ * 10000)

plt.figure(figsize=(7, 5))
p = plt.subplot()
p.plot(x, y_truth, color='gray', label='ground truth')
p.plot(x2, y_plain, color='r', markersize=2, label='No Regularization')
p.plot(x2, y_lasso, color='g',  markersize=2, label='Lasso')
p.plot(x2, y_ridge, color='b',  markersize=2, label='Ridge')
p.plot(x, y_data_plot, '.', color='k')
p.legend()
p.set_ylim(-1, 1)

plt.show()
