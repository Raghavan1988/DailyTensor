"""
Tensor-Parallel Multi-Head Attention (Megatron-style), NumPy simulation.

Multi-head attention is "embarrassingly parallel" across heads: each head runs
its own softmax-attention on its own slice of Q/K/V and never looks at another
head's data.  Megatron exploits this to split the HEADS across `P` devices,
which mirrors the MLP pattern from program 01:

  * QKV projection is COLUMN-parallel:  device i builds Q, K, V for only its
    subset of heads (H/P heads).  No communication.
  * Each device runs full attention for its local heads.  No communication.
  * Output projection W_o is ROW-parallel:  the per-head context vectors form a
    column-slice of the concatenated heads, so W_o is split by rows to match.
    Each device produces a PARTIAL [seq, d_model] output ...
  * ... and a single all-reduce sums the partials into the final result.

We verify the P-way head split reproduces single-device attention exactly.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V):
    # Q, K, V: [n_heads, seq, d_head] -> scaled dot-product attention per head
    d_head = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)   # [n_heads, seq, seq]
    return softmax(scores, axis=-1) @ V                   # [n_heads, seq, d_head]


def project_to_heads(X, W, n_heads):
    # X:[seq, d_model], W:[d_model, n_heads*d_head] -> [n_heads, seq, d_head]
    seq = X.shape[0]
    d_head = W.shape[1] // n_heads
    proj = X @ W                                          # [seq, n_heads*d_head]
    return proj.reshape(seq, n_heads, d_head).transpose(1, 0, 2)


def single_device_mha(X, Wq, Wk, Wv, Wo, H):
    Q, K, V = (project_to_heads(X, W, H) for W in (Wq, Wk, Wv))
    ctx = attention(Q, K, V)                              # [H, seq, d_head]
    seq = X.shape[0]
    ctx = ctx.transpose(1, 0, 2).reshape(seq, -1)         # concat heads -> [seq, d_model]
    return ctx @ Wo


def tensor_parallel_mha(X, Wq, Wk, Wv, Wo, H, P):
    assert H % P == 0, "heads must divide evenly across ranks"
    hpr = H // P                                          # heads per rank
    d_head = X.shape[1] // H
    partials = []
    for i in range(P):
        # column-parallel QKV: keep only the columns feeding this rank's heads
        cols = slice(i * hpr * d_head, (i + 1) * hpr * d_head)
        Q = project_to_heads(X, Wq[:, cols], hpr)
        K = project_to_heads(X, Wk[:, cols], hpr)
        V = project_to_heads(X, Wv[:, cols], hpr)
        ctx = attention(Q, K, V)                          # [hpr, seq, d_head]
        ctx = ctx.transpose(1, 0, 2).reshape(X.shape[0], -1)   # [seq, hpr*d_head]
        # row-parallel output projection: the rows of W_o for this rank's heads
        partials.append(ctx @ Wo[cols, :])                # [seq, d_model] partial
    return sum(partials)                                  # all-reduce


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq, d_model, H = 5, 16, 8
    X = rng.standard_normal((seq, d_model))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d_model, d_model)) for _ in range(4))

    reference = single_device_mha(X, Wq, Wk, Wv, Wo, H)
    for P in (1, 2, 4, 8):
        out = tensor_parallel_mha(X, Wq, Wk, Wv, Wo, H, P)
        err = np.abs(out - reference).max()
        print(f"P={P}: max abs diff vs single device = {err:.2e}")
    print("\nSplitting heads across ranks reproduces single-device attention.")
