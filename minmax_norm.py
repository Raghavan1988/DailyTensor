import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0, scale per column.
    Return np.ndarray (float).
    """
    X = np.array(X, dtype=float)

    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    ans = (X - x_min) / (x_max - x_min + eps)
    return ans
