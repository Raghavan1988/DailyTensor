"""
Problem: Softmax (NumPy)
------------------------
Implement the softmax function using NumPy.

    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

Requirements:
  1. Accept a Python list, a 1D NumPy array, or a 2D NumPy array as input.
  2. For 2D input, apply softmax along the given `axis` (default: last axis,
     i.e. row-wise for a (batch, features) matrix). The output shape must
     match the input shape.
  3. Be numerically stable: exp(1000) overflows. Use the standard trick of
     subtracting the max value along `axis` before exponentiating.
     (Hint: this does NOT change the result mathematically.)
  4. Return a float64 NumPy array. Each slice along `axis` must sum to 1.

Examples:
    >>> softmax([1.0, 2.0, 3.0])
    array([0.09003057, 0.24472847, 0.66524096])

    >>> softmax([[1, 2, 3], [1, 2, 3]], axis=1).sum(axis=1)
    array([1., 1.])

    >>> softmax([1000.0, 1000.0, 1000.0])      # must NOT overflow
    array([0.33333333, 0.33333333, 0.33333333])
"""

import numpy as np


def softmax(x, axis=-1):
    # Convert input (list / 1D / 2D) to a float64 NumPy array so math is vectorized and precise.
    x = np.asarray(x, dtype=np.float64)
    # Subtract the max along `axis` for numerical stability; keepdims=True preserves shape for broadcasting.
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    # Exponentiate the shifted values; max of x_shifted is 0, so exp() can't overflow.
    exps = np.exp(x_shifted)
    # Normalize by the sum along `axis` so each slice sums to 1; keepdims=True lets it broadcast across rows/cols.
    return exps / np.sum(exps, axis=axis, keepdims=True)
