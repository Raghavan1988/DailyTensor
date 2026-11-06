import numpy as np


def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Returns a Python float.
    """
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch: {X.shape} vs {Y.shape}")
    return float(np.sqrt(np.sum(np.square(X - Y))))
