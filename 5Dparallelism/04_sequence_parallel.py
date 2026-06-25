"""
Sequence Parallelism (Megatron-style), NumPy simulation.

Plain tensor parallelism (TP) keeps LayerNorm / residual / dropout *replicated*
on every device -- each one stores the full [seq, d_model] activation.  Sequence
parallelism (SP) instead shards those regions along the SEQUENCE dimension, so
each of the P devices stores only [seq/P, d_model].  Same numbers, much less
activation memory.

The interesting part is how data moves between the two kinds of region:

    SP region (seq-sharded)   --g-->     TP region (hidden-sharded)
                              <--g_bar--

  * g      (forward) = ALL-GATHER along sequence:     [s/P, h] -> [s, h]
  * g_bar  (forward) = REDUCE-SCATTER along sequence: sum the TP partials AND
                       re-shard them back to sequence: P*[s, h] -> [s/P, h]

This is "free" because of the identity that underlies all of it:

        ALL-REDUCE  ==  REDUCE-SCATTER  followed by  ALL-GATHER

so SP moves the same number of bytes as TP's all-reduce, just split into two
cheaper halves -- while storing less activation memory in between.

This file demonstrates BOTH: the all-reduce identity, and a full SP transformer
block whose output matches the single-device block.
"""

import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layernorm(x, eps=1e-5):
    # normalises over the feature dim, independently per token (row) -> sharding
    # along the sequence dim is therefore exact, no cross-shard stats needed.
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ---- collective primitives, faked over a Python list of P rank-tensors ----
def all_gather_seq(shards):
    """Each rank's [s/P, h] shard -> the full [s, h], visible to every rank."""
    return np.concatenate(shards, axis=0)


def reduce_scatter_seq(partials, P):
    """Sum the P partial [s, h] tensors, then scatter the rows into P shards."""
    total = sum(partials)                 # the "reduce" (sum across ranks)
    return np.split(total, P, axis=0)     # the "scatter" (re-shard the sequence)


def all_reduce(partials):
    return sum(partials)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = 4
    seq, d_model, d_ff = 8, 16, 64
    assert seq % P == 0
    X = rng.standard_normal((seq, d_model))
    A = rng.standard_normal((d_model, d_ff))   # column-parallel weight
    B = rng.standard_normal((d_ff, d_model))   # row-parallel weight

    # ---------- 1) the identity that makes SP free ----------
    partials = [rng.standard_normal((seq, d_model)) for _ in range(P)]
    via_allreduce = all_reduce(partials)
    via_rs_then_ag = all_gather_seq(reduce_scatter_seq(partials, P))
    print("all-reduce  ==  reduce-scatter + all-gather :",
          np.allclose(via_allreduce, via_rs_then_ag))

    # ---------- 2) reference: one device, no parallelism ----------
    def block(X):
        h = layernorm(X)
        h = gelu(h @ A) @ B
        return X + h                           # residual connection
    reference = block(X)

    # ---------- 3) the same block run with TP + SP ----------
    shard = d_ff // P
    A_sh = [A[:, i * shard:(i + 1) * shard] for i in range(P)]   # column split
    B_sh = [B[i * shard:(i + 1) * shard, :] for i in range(P)]   # row split

    # SP region: each rank holds only its sequence slice and LayerNorms it
    X_sp = np.split(X, P, axis=0)                       # [s/P, h] per rank
    normed_sp = [layernorm(x) for x in X_sp]

    # g: all-gather to the full sequence so the TP matmuls can run
    normed_full = all_gather_seq(normed_sp)             # [s, h] on every rank

    # TP region: column-parallel -> GeLU -> row-parallel -> partial [s, h]
    tp_partials = [gelu(normed_full @ A_sh[i]) @ B_sh[i] for i in range(P)]

    # g_bar: reduce-scatter back to sequence shards (sum, then re-shard)
    mlp_sp = reduce_scatter_seq(tp_partials, P)         # [s/P, h] per rank

    # residual add stays sequence-sharded; gather only so we can compare
    out_sp = [X_sp[i] + mlp_sp[i] for i in range(P)]
    out_full = all_gather_seq(out_sp)

    print("TP+SP block == single-device block        :",
          np.allclose(out_full, reference))
    print("\nIn the SP regions each rank stored only seq/P =", seq // P,
          "rows (vs", seq, "for plain TP).")
