import numpy as np


def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError(f"expected 1D inputs, got shapes {X.shape} and {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError(f"length mismatch: {X.shape[0]} vs {Y.shape[0]}")
    # NumPy arrays expose a .dot() method equivalent to np.dot(X, Y).
    return float(X.dot(Y))
