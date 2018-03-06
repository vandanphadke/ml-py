import numpy as np

def add_ones_col(mat):
    n,m = mat.shape
    one = np.ones((n,1))
    return np.hstack((one, mat))
