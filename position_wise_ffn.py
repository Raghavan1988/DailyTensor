"""
Problem: Position-wise Feed-Forward Network (NumPy)
---------------------------------------------------
Every Transformer block contains a 2-layer MLP applied INDEPENDENTLY to
each position (the same weights are reused across positions):

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

This per-position non-linearity is where most of a Transformer's
parameters and compute live.

Requirements:
  1. x has shape (batch, seq, d_model).
  2. W1: (d_model, d_ff), b1: (d_ff,)
  3. W2: (d_ff, d_model), b2: (d_model,)
  4. Apply ReLU between the two linear layers.
  5. Return an array of shape (batch, seq, d_model).

Example:
    >>> x  = np.zeros((1, 3, 4))
    >>> W1 = np.ones((4, 8)); b1 = np.zeros(8)
    >>> W2 = np.ones((8, 4)); b2 = np.zeros(4)
    >>> position_wise_ffn(x, W1, b1, W2, b2).shape
    (1, 3, 4)
"""

import numpy as np


def position_wise_ffn(x, W1, b1, W2, b2):
    # Convert input to float64 — biases/weights broadcast cleanly during matmul.
    x = np.asarray(x, dtype=np.float64)
    W1, b1, W2, b2 = (np.asarray(a, dtype=np.float64) for a in (W1, b1, W2, b2))
    # Validate shapes up front so mistakes surface with a clear message instead of a matmul error.
    d_model = x.shape[-1]
    if W1.ndim != 2 or W1.shape[0] != d_model:
        raise ValueError(f"W1 must have shape ({d_model}, d_ff), got {W1.shape}")
    d_ff = W1.shape[1]
    if b1.shape != (d_ff,):
        raise ValueError(f"b1 must have shape ({d_ff},), got {b1.shape}")
    if W2.shape != (d_ff, d_model):
        raise ValueError(f"W2 must have shape ({d_ff}, {d_model}), got {W2.shape}")
    if b2.shape != (d_model,):
        raise ValueError(f"b2 must have shape ({d_model},), got {b2.shape}")
    # First linear layer expands the feature dim from d_model to d_ff (typically 4x larger).
    hidden = x @ W1 + b1
    # ReLU introduces the non-linearity — without it the two linear layers would collapse to one.
    hidden = np.maximum(hidden, 0.0)
    # Second linear layer projects back from d_ff to d_model.
    return hidden @ W2 + b2
