
"""
Problem: Clip Values (NumPy)
----------------------------
Clamp every element of an array into the range [low, high]:

    clip(x, low, high)_i = min(max(x_i, low), high)

Requirements:
  1. Accept a Python list, 1D array, or 2D array.
  2. Return a float64 NumPy array with the same shape as the input.
  3. Raise ValueError if low > high.

Examples:
    >>> clip_values([-2, 0, 3, 7, 10], low=0, high=5)
    array([0., 0., 3., 5., 5.])

    >>> clip_values([[1, 5], [-1, 9]], low=0, high=4)
    array([[1., 4.],
           [0., 4.]])
"""

import numpy as np


def clip_values(x, low, high):
    # Validate the range — clipping with low > high would silently produce nonsense.
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    # Convert input to a float64 NumPy array so the result is consistent across input types.
    x = np.asarray(x, dtype=np.float64)
    # np.clip clamps every element to [low, high] in one vectorized call.
    return np.clip(x, low, high)
