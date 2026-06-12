import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    X = np.asarray(x)
    Y = np.asarray(y)
    return X.dot(Y)
