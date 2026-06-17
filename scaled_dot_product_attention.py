"""
Problem: Scaled Dot-Product Attention (NumPy)
---------------------------------------------
Implement the core attention mechanism from "Attention Is All You Need":

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Requirements:
  1. Accept Q, K, V as 2D arrays of shapes (n_q, d_k), (n_k, d_k), (n_k, d_v).
  2. Scale dot products by 1/sqrt(d_k) BEFORE the softmax (prevents the
     dot products from growing large in magnitude and pushing softmax
     into regions with vanishing gradients).
  3. Apply softmax along the last axis so each row of attention weights
     sums to 1.
  4. Return the output array of shape (n_q, d_v).

Example:
    >>> Q = np.array([[1.0, 0.0]])
    >>> K = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> V = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> scaled_dot_product_attention(Q, K, V).shape
    (1, 2)
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V):
    # Convert inputs to float64 arrays so matmul and division are precise.
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    # d_k is the key/query dimension — used to scale the dot products.
    d_k = Q.shape[-1]
    # Compute raw attention scores: each query scored against every key, then scaled.
    scores = Q @ K.T / np.sqrt(d_k)
    # Numerically stable softmax along the last axis — subtract row max to prevent overflow.
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    # Weighted sum of value vectors gives the attended output for each query.
    return weights @ V
