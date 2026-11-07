"""
Problem: Sinusoidal Positional Encoding (NumPy)
-----------------------------------------------
Transformers have no recurrence or convolution, so position information
must be injected explicitly. Compute the sinusoidal positional encoding
matrix from "Attention Is All You Need":

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Requirements:
  1. Accept integer arguments `seq_len` and `d_model` (odd d_model is allowed;
     the final column is then a lone sin term).
  2. Return a NumPy array of shape (seq_len, d_model).
  3. Even-index columns use sin; odd-index columns use cos.

Example:
    >>> positional_encoding(4, 8).shape
    (4, 8)

    >>> pe = positional_encoding(1, 4)
    >>> pe[0, 0], pe[0, 1]              # sin(0)=0, cos(0)=1
    (0.0, 1.0)
"""

import numpy as np


def positional_encoding(seq_len, d_model):
    # Column vector of positions: 0, 1, ..., seq_len-1.
    positions = np.arange(seq_len, dtype=np.float64).reshape(-1, 1)
    if seq_len < 0 or d_model <= 0:
        raise ValueError(f"need seq_len >= 0 and d_model > 0, got {seq_len}, {d_model}")
    # i indexes the ceil(d_model/2) sin/cos pairs; for odd d_model the last pair has no cos.
    i = np.arange((d_model + 1) // 2, dtype=np.float64)
    # Denominator term 10000^(2i / d_model) — shared between each sin/cos pair.
    denom = np.power(10000.0, (2.0 * i) / d_model)
    # Broadcast: (seq_len, 1) / (ceil(d_model/2),) -> angle matrix.
    angles = positions / denom
    # Interleave sin and cos: sin into even columns, cos into odd columns.
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, : d_model // 2])
    return pe
