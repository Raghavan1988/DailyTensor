import numpy as np


def euclidean_distance(x, y, axis=None):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Returns a Python float.
    """
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch: {X.shape} vs {Y.shape}")
    # axis=None reduces everything to a scalar; pass axis=-1 for row-wise
    # distances between batches of vectors.
    d = np.sqrt(np.sum(np.square(X - Y), axis=axis))
    return float(d) if np.ndim(d) == 0 else d
