"""
Problem: L2 Normalize (NumPy)
-----------------------------
Scale a vector (or each row of a matrix) to have L2 norm = 1:

    normalize(x) = x / ||x||_2

Requirements:
  1. Accept a 1D or 2D input (list or NumPy array).
  2. For 2D input, normalize each row independently.
  3. If a vector has norm 0, return all zeros for it (do NOT divide by zero).
  4. Return a float64 NumPy array with the same shape as the input.

Examples:
    >>> l2_normalize([3.0, 4.0])
    array([0.6, 0.8])

    >>> l2_normalize([[3, 4], [0, 0], [1, 0]])
    array([[0.6, 0.8],
           [0. , 0. ],
           [1. , 0. ]])
"""

import numpy as np


def l2_normalize(x):
    # Convert input to a float64 array so the division produces floats.
    x = np.asarray(x, dtype=np.float64)
    # Compute L2 norm along the last axis; keepdims=True lets it broadcast for division.
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # Replace any zero norms with 1 to avoid division by zero (we'll zero those rows out below).
    safe_norms = np.where(norms == 0, 1.0, norms)
    # Divide every element by its row's norm (or 1 if the row was all zeros).
    return x / safe_norms
