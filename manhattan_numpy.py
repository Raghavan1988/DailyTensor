import numpy as np


def manhattan(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y:

        d(x, y) = sum_i |x_i - y_i|

    Must return a float.
    """
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch: {X.shape} vs {Y.shape}")
    return float(np.sum(np.abs(X - Y)))
