import numpy as np


def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0, scale per column.
    Return np.ndarray (float).

    Constant slices (max - min <= eps) map to 0 instead of dividing by
    ~zero. Non-constant slices are scaled exactly, so the max lands on 1.0.
    """
    X = np.array(X, dtype=float)

    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    value_range = x_max - x_min
    constant = value_range <= eps
    # Divide by 1 where the slice is constant; (X - x_min) is already 0 there.
    safe_range = np.where(constant, 1.0, value_range)
    return (X - x_min) / safe_range
