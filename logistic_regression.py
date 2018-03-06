import math
import numpy as np
from utils import *

from sklearn.datasets import make_classification

X, Y = make_classification(n_samples=10, n_features=2,
        n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)

# Getting correct coefficients from
# sklearn logisticregression for comparision
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, Y)
print(lr.coef_)

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-1.0*z))

def vsigmoid(zarr):
    func = np.vectorize(sigmoid)
    return func(zarr)

def hypothesis(theta, X):
    tmp = np.matmul(X, theta)
    return vsigmoid(tmp)

def cost(X, Y, theta):
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        if Y[i] == 1:
            total_cost += hypothesis(theta, X[i])
        else:
            total_cost += (1 - hypothesis(theta, X[i]))
    return (1.0/m)*total_cost

def gradient_descent_logit(X, Y, alpha=0.1, n_iters=1000):
    n,m = X.shape
    theta = np.random.rand(m, 1)
    for i in range(n_iters):
        cost_old = cost(X, Y, theta)
        h = hypothesis(theta, X)
        h_y = np.subtract(h, np.reshape(Y, (n, 1)))
        h_y_x = (np.matmul(X.transpose(), h_y))
        theta_new = np.subtract(theta, (alpha/n)*h_y_x)
        cost_new = cost(X, Y, theta_new)
        theta = theta_new
    return theta

def classify_logit(X_train, Y_train, X_test):
    coeff = gradient_descent_logit(X_train, Y_train)
    probabilities = hypothesis(coeff, X_test)
    predictions = list(map(lambda x: 1 if x > 0.5 else 0, probabilities))
    return predictions

Xtest = np.random.rand(5, 2)
classify_logit(X, Y, Xtest)
