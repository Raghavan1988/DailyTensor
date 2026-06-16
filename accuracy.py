"""
Problem: Classification Accuracy (NumPy)
----------------------------------------
Compute the fraction of correct predictions:

    accuracy = (# of i where y_true[i] == y_pred[i]) / N

Requirements:
  1. Accept lists or 1D NumPy arrays of equal length.
  2. Return a single Python float in [0.0, 1.0].
  3. Raise ValueError if shapes don't match.
  4. Return 0.0 if the input is empty.

Examples:
    >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    0.75

    >>> accuracy([2, 2, 2], [2, 2, 2])
    1.0

    >>> accuracy([], [])
    0.0
"""

import numpy as np


def accuracy(y_true, y_pred):
    # Convert both inputs to NumPy arrays so the equality comparison is vectorized.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Reject mismatched shapes — silent broadcasting on (N,) vs (M,) would be a bug.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    # Empty input has no predictions to score — return 0.0 instead of NaN from 0/0.
    if y_true.size == 0:
        return 0.0
    # Element-wise equality produces a boolean array; mean of booleans = fraction of True.
    return float(np.mean(y_true == y_pred))
