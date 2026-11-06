"""
Problem: Mean Absolute Error (NumPy)
------------------------------------
Compute the mean absolute error between true values and predictions:

    MAE = mean( |y_true - y_pred| )

Unlike MSE, MAE penalizes errors linearly, so it is less sensitive to
outliers.

Requirements:
  1. Accept lists or NumPy arrays of equal shape.
  2. Return a single Python float.
  3. Raise ValueError if shapes don't match.

Examples:
    >>> mae([1, 2, 3], [1, 2, 3])
    0.0

    >>> mae([1, 2, 3], [2, 4, 6])
    2.0
"""

import numpy as np


def mae(y_true, y_pred):
    # Convert both inputs to float64 NumPy arrays so subtraction is vectorized.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Guard against silent broadcasting bugs by requiring matching shapes up front.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    # Element-wise absolute difference, then the mean over all elements.
    return float(np.mean(np.abs(y_true - y_pred)))
