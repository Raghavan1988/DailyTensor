import numpy as np

def manhattan(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    X = np.array(x)
    Y = np.array(y)
    dist = np.sum(np.abs(X-Y)).item()


    return dist
