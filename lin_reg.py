import numpy as np
import utils

X = np.round(np.random.rand(100, 3), 3)
Y = 3 + X[:, 0] + 5*X[:, 1] + 3*X[:, 2]

def solve_normal_eqn(X, y):
    Xnew = add_ones_col(X)
    return np.matmul(np.linalg.inv(np.matmul(Xnew.transpose(), Xnew)),
                     np.matmul(Xnew.transpose(), y))

def compute_cost(X, y, theta):
    n,m = X.shape
    h = np.round(np.matmul(X, theta), 3)
    sq_errors = np.square(np.subtract(h, y))
    return (1.0/(2.0*m))*(sq_errors.sum())

def solve_gradient_descent(X, y, alpha=0.01, n_iters=5000):
    X = add_ones_col(X)
    m,n = X.shape
    theta = np.random.rand(n, 1)
    epsilon = 100
    for i in range(1, n_iters):
        cost_old = compute_cost(X, Y, theta)
        h = np.matmul(X, theta)
        h_y = np.subtract(h, Y.reshape((100, 1)))
        h_y_x = (np.matmul(X.transpose(), h_y)).transpose()
        theta_new = np.subtract(theta, alpha*h_y_x.transpose())
        cost_new = compute_cost(X, Y, theta_new)
        epsilon = cost_old - cost_new
        theta = theta_new
    return theta

solve_gradient_descent(X, Y)
