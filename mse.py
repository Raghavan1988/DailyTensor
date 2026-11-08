"""
Problem: Mean Squared Error (NumPy)
-----------------------------------
Compute the mean squared error between true values and predictions:

    MSE = mean( (y_true - y_pred) ** 2 )

Requirements:
  1. Accept lists or NumPy arrays of equal shape.
  2. Return a single Python float.
  3. Raise ValueError if shapes don't match.
  4. Pass squared=False to get the root mean squared error (RMSE),
     which is in the same units as the targets.

Examples:
    >>> mse([1, 2, 3], [1, 2, 3])
    0.0

    >>> mse([1, 2, 3], [2, 4, 6])
    4.666666666666667

    >>> mse([0, 0], [3, 4], squared=False)
    3.5355339059327378
"""

import numpy as np


def mse(y_true, y_pred, squared=True):
    # Convert both inputs to float64 NumPy arrays so subtraction and squaring are vectorized.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Guard against silent broadcasting bugs by requiring matching shapes up front.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    # Element-wise difference, square it, then take the mean over all elements.
    value = float(np.mean((y_true - y_pred) ** 2))
    return value if squared else float(np.sqrt(value))
