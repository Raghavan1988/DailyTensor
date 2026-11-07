import numpy as np


def relu(x):
    """
    ReLU activation: relu(x) = max(x, 0), applied element-wise.

    Accepts scalars, lists or arrays of any shape; returns a float64 array.

    Example:
        >>> relu([-2, 0, 3])
        array([0., 0., 3.])
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0)


def relu_grad(x):
    """
    Derivative of ReLU w.r.t. its input: 1 where x > 0, else 0.

    ReLU is not differentiable at 0; we use the common subgradient choice 0.

    Example:
        >>> relu_grad([-2, 0, 3])
        array([0., 0., 1.])
    """
    x = np.asarray(x, dtype=np.float64)
    return (x > 0).astype(np.float64)
