"""
Tensor-Parallel MLP (Megatron-style), simulated on one machine with NumPy.

The transformer feed-forward block computes:

        Y = GeLU(X @ A) @ B

where A is [d_model, d_ff] and B is [d_ff, d_model].

Megatron splits this across `P` devices with only ONE communication step,
by choosing the split direction of each weight cleverly:

  * A is split by COLUMNS  ->  A = [A_0 | A_1 | ... | A_{P-1}]
        Device i gets A_i of shape [d_model, d_ff/P] and computes its own
        slice of the hidden activation.  GeLU is element-wise, so it can be
        applied to each slice independently -- NO communication needed.

  * B is split by ROWS     ->  B = [B_0; B_1; ... ; B_{P-1}]
        Device i holds B_i of shape [d_ff/P, d_model] and multiplies it with
        its hidden slice, producing a PARTIAL output of the full [.., d_model]
        shape (a partial *sum*, since a row-split matmul is a sum of products).

  * The partial outputs are summed across devices (an "all-reduce").

So the only communication is a single all-reduce at the very end.  This file
fakes the P devices with a Python list and fakes the all-reduce with sum(...),
then checks the result matches the single-device computation exactly.
"""

import numpy as np


def gelu(x):
    # tanh approximation of the Gaussian Error Linear Unit
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def single_device_mlp(X, A, B):
    """The reference computation we want to reproduce in parallel."""
    return gelu(X @ A) @ B


def tensor_parallel_mlp(X, A, B, P):
    """Simulate the same MLP split across P tensor-parallel ranks."""
    d_ff = A.shape[1]
    assert d_ff % P == 0, "hidden size must divide evenly across ranks"
    shard = d_ff // P

    # --- "scatter" the weights to each rank (done once, at model-load time) ---
    A_shards = [A[:, i * shard:(i + 1) * shard] for i in range(P)]   # column split
    B_shards = [B[i * shard:(i + 1) * shard, :] for i in range(P)]   # row split

    # --- each rank computes independently; no communication in this loop ---
    partial_outputs = []
    for i in range(P):
        Z_i = gelu(X @ A_shards[i])     # [.., d_ff/P]  local hidden slice
        Y_i = Z_i @ B_shards[i]         # [.., d_model] partial output
        partial_outputs.append(Y_i)

    # --- all-reduce: sum the partial outputs across all ranks ---
    Y = sum(partial_outputs)
    return Y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch, d_model, d_ff = 4, 8, 32
    X = rng.standard_normal((batch, d_model))
    A = rng.standard_normal((d_model, d_ff))
    B = rng.standard_normal((d_ff, d_model))

    reference = single_device_mlp(X, A, B)
    for P in (1, 2, 4, 8):
        out = tensor_parallel_mlp(X, A, B, P)
        err = np.abs(out - reference).max()
        print(f"P={P}: max abs diff vs single device = {err:.2e}")
    print("\nEvery sharding matches the single-device result -> TP is exact.")
