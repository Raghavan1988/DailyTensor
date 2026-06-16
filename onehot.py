"""
Problem: One-Hot Encoding (NumPy)
---------------------------------
Convert a 1D array of integer class labels into a 2D one-hot encoded matrix.

Given:
  - `labels`: a list or 1D NumPy array of N integer class indices in [0, num_classes).
  - `num_classes`: total number of distinct classes.

Return:
  - A 2D NumPy array of shape (N, num_classes) and dtype float64, where
    row i is all zeros except for a 1.0 at column labels[i].

Requirements:
  1. Accept either a Python list or a NumPy array as input.
  2. Use vectorized NumPy indexing — NO Python `for` loops over rows.
  3. Output shape must be (len(labels), num_classes).

Examples:
    >>> one_hot([0, 2, 1, 2], num_classes=3)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.]])

    >>> one_hot([3, 0], num_classes=5).shape
    (2, 5)

    >>> one_hot([], num_classes=4).shape
    (0, 4)
"""

import numpy as np


def one_hot(labels, num_classes):
    # Convert input (list or array) to a 1D int64 NumPy array so it can index into rows safely.
    labels = np.asarray(labels, dtype=np.int64)
    # Pre-allocate the output matrix of zeros with shape (N, num_classes) and dtype float64.
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    # Use fancy indexing: for every row i, set column labels[i] to 1.0 in a single vectorized op.
    out[np.arange(labels.shape[0]), labels] = 1.0
    # Return the populated one-hot matrix.
    return out
