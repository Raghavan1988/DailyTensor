### https://www.tensortonic.com/problems/split-heads-numpy
"""
Problem: Multi-Head Reshape — split_heads (NumPy)
-------------------------------------------------
Multi-head attention splits an embedding of size d_model into `num_heads`
parallel subspaces of size d_head = d_model / num_heads, runs attention
in each subspace independently, then concatenates. Implement the
reshape + transpose that splits the last dimension into separate heads:

    (batch, seq, d_model) -> (batch, num_heads, seq, d_head)

Putting `num_heads` next to `batch` lets the rest of the multi-head
attention pipeline treat each head as just another "batch" element.

Requirements:
  1. Accept x of shape (batch, seq, d_model) and integer `num_heads`.
  2. d_model must be divisible by num_heads; raise ValueError otherwise.
  3. Return an array of shape (batch, num_heads, seq, d_head).

Example:
    >>> x = np.zeros((2, 5, 8))
    >>> split_heads(x, num_heads=4).shape
    (2, 4, 5, 2)
"""

import numpy as np
def split_heads(x, num_heads):
    # Convert input to a float64 array so downstream math is consistent.
    x = np.asarray(x, dtype=np.float64)
    batch, seq, d_model = x.shape
    # Each head must get a clean integer slice of the feature dimension.
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
    d_head = d_model // num_heads
    # Split the feature axis into (num_heads, d_head) — no data movement, just reshape.
    x = x.reshape(batch, seq, num_heads, d_head)
    # Move the head axis next to batch so each head sees an independent (seq, d_head) slice.
    return x.transpose(0, 2, 1, 3)
