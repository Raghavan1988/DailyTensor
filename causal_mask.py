"""
Problem: Causal (Look-Ahead) Attention Mask (NumPy)
---------------------------------------------------
Decoder-only Transformers (like GPT) must NOT attend to future tokens at
training time. Build the additive mask that, when added to the attention
score matrix BEFORE softmax, zeros out the upper-triangular attention
weights:

    mask[i, j] = 0          if j <= i   (allowed: past + current)
    mask[i, j] = -infinity  if j > i    (forbidden future -> becomes 0 after softmax)

Adding -inf to a score makes exp(score) = 0, which is why this works.

Requirements:
  1. Accept integer `seq_len`.
  2. Return a NumPy float array of shape (seq_len, seq_len).
  3. Lower triangle (including diagonal) = 0.0; strict upper triangle = -inf.

Example:
    >>> causal_mask(3)
    array([[  0., -inf, -inf],
           [  0.,   0., -inf],
           [  0.,   0.,   0.]])
"""

import numpy as np


def causal_mask(seq_len):
    # Start with an all-zeros matrix — "everything allowed" by default.
    mask = np.zeros((seq_len, seq_len), dtype=np.float64)
    # np.triu with k=1 marks the STRICT upper triangle (future positions only).
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    # Block future positions by setting their score addend to -inf.
    mask[future] = -np.inf
    return mask
