"""
Problem: Layer Normalization (NumPy)
------------------------------------
Layer norm — the normalization sandwiched around every Transformer
sub-layer — normalizes each sample independently across its feature axis:

    LN(x) = (x - mean(x)) / sqrt(var(x) + eps)

Unlike batch norm, layer norm does NOT mix statistics across the batch,
which makes it well-suited to variable-length sequences.

Requirements:
  1. Accept a 2D input of shape (batch, features).
  2. Compute mean and variance along the LAST axis (per row).
  3. Add a small epsilon (default 1e-5) inside the sqrt to avoid
     divide-by-zero when a row is constant.
  4. Return a float64 array of the same shape. Each row should have
     ~0 mean and ~1 standard deviation.

Example:
    >>> x = np.array([[1.0, 2.0, 3.0, 4.0]])
    >>> y = layer_norm(x)
    >>> np.round(y.mean(axis=-1), 8)
    array([0.])
"""

import numpy as np


def layer_norm(x, eps=1e-5):
    # Convert input to float64 so the division returns a precise float array.
    x = np.asarray(x, dtype=np.float64)
    # Mean across the feature axis for each sample; keepdims=True lets it broadcast.
    mean = np.mean(x, axis=-1, keepdims=True)
    # Variance across the feature axis for each sample.
    var = np.var(x, axis=-1, keepdims=True)
    # Normalize each row to zero mean / unit variance; eps stabilizes when var ~ 0.
    return (x - mean) / np.sqrt(var + eps)
