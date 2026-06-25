"""Sequence Parallelism + Tensor-Parallel MLP — the all_gather ⇄ reduce_scatter duality.

CONCEPT
-------
Megatron-LM "Reducing Activation Recomputation in Large Transformer Models" observes that
a Transformer block has two kinds of regions:
  1. TENSOR-PARALLEL regions  — the two MLP matmuls X@A (column-parallel) and H@B (row-parallel).
  2. "the rest"               — LayerNorm + dropout, which are element-wise along hidden and
                                 fully parallel along the SEQUENCE dimension.
Plain TP (program 03) keeps the LayerNorm input REPLICATED on every rank: each rank stores the
*full* (seq, hidden) activation. That is wasteful — LayerNorm doesn't need a replicated tensor.
Sequence Parallelism (SP) shards those "rest" regions along the SEQUENCE axis instead, so each
rank stores only seq/W rows. The activation memory of the LayerNorm region drops by a factor W.

WHICH AXIS IS SHARDED
---------------------
  - LayerNorm / dropout region : sharded along the SEQUENCE axis (each rank holds seq/W rows).
  - matmul weights A, B        : sharded as in TP (A column-parallel, B row-parallel along hidden).

WHICH COLLECTIVE & WHY (the conjugate pair g and g_bar)
-------------------------------------------------------
To move between the two regions we need two transition operators that are each other's conjugate:
  - g     = all_gather(SEQ)      entering the TP region: the column-parallel matmul X@A needs the
                                 WHOLE sequence as input, so we gather the seq-shards into full X.
  - g_bar = reduce_scatter(SEQ)  leaving the TP region: the row-parallel matmul produces PARTIAL
                                 sums that must be summed (like TP's all_reduce) AND we want to land
                                 back in seq-sharded form — reduce_scatter does BOTH in one call.
KEY INSIGHT: plain TP used ONE all_reduce here. all_reduce == all_gather ∘ reduce_scatter, so SP's
two collectives move the *same total bytes* as TP's one all_reduce — communication volume is
UNCHANGED — but SP keeps the LayerNorm-region activation sharded (1/W memory). Free memory win.

COMMUNICATION COST PER FORWARD
------------------------------
2 collectives: 1 all_gather (enter TP region) + 1 reduce_scatter (leave TP region).
Same bytes on the wire as plain TP's single all_reduce.

SIMULATION NOTE
---------------
We simulate W GPUs in ONE process; each 'rank' only ever touches its own shard.
"""

import numpy as np

np.random.seed(0)

W = 4  # number of simulated GPUs ("world size")


# ---------------------------------------------------------------------------
# Collectives — tiny explicit stand-ins for torch.distributed / NCCL ops.
# ---------------------------------------------------------------------------
def all_gather(shards, axis):
    """Concatenate every rank's shard along 'axis'; result identical on all ranks (== dist.all_gather)."""
    return np.concatenate(shards, axis=axis)


def reduce_scatter(shards, axis):
    """SUM the shards, then SPLIT the sum along 'axis' so each rank keeps only its slice (== dist.reduce_scatter)."""
    summed = np.sum(shards, axis=0)            # the "reduce" half (identical to all_reduce's sum)
    return split(summed, W, axis)              # the "scatter" half (back to seq-shards)


def split(x, W, axis):
    """Helper: split x into W equal shards along 'axis' (the inverse of all_gather)."""
    return np.split(x, W, axis=axis)


# ---------------------------------------------------------------------------
# Math kernels.
# ---------------------------------------------------------------------------
def layernorm(x):
    """Standard LayerNorm over the last (hidden) axis — element-wise per sequence row."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)


def gelu(x):
    """Tanh approximation of GeLU (the MLP nonlinearity)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# Problem setup.  seq must divide evenly by W; hidden divides by W for the TP weights.
# ---------------------------------------------------------------------------
seq, hidden, ffn = 8, 6, 12
X = np.random.randn(seq, hidden)               # full (replicated-conceptually) input
A = np.random.randn(hidden, ffn)               # first MLP weight (gets COLUMN-parallel sharding)
B = np.random.randn(ffn, hidden)               # second MLP weight (gets ROW-parallel sharding)

# SEQUENCE-PARALLEL input: X is sharded along the SEQUENCE axis -> X_0..X_{W-1}, each seq/W rows.
X_seq_shards = split(X, W, axis=0)
# TP weight shards: A column-parallel (split ffn), B row-parallel (split ffn rows).
A_shards = split(A, W, axis=1)
B_shards = split(B, W, axis=0)

print(f"world size W = {W}")
print(f"full activation row count = {seq}; SP stores {seq // W} rows/rank along SEQUENCE axis")
print(f"per-rank SP activation shape = {X_seq_shards[0].shape}  vs  full = {X.shape}  "
      f"(memory for LayerNorm region: 1/{W})")
print(f"per-rank A (col-parallel) = {A_shards[0].shape}; per-rank B (row-parallel) = {B_shards[0].shape}")


# ---------------------------------------------------------------------------
# SEQUENCE-PARALLEL forward pass.
# ---------------------------------------------------------------------------
# Region 1 (SEQUENCE-PARALLEL): LayerNorm on each rank's OWN seq-shard only — 1/W memory.
ln_seq_shards = [layernorm(xs) for xs in X_seq_shards]

# g = all_gather(SEQ): gather the seq-sharded, layer-normed activations into the FULL sequence
#     that the column-parallel matmul X@A needs (input is now replicated across ranks).
ln_full = all_gather(ln_seq_shards, axis=0)    # shape (seq, hidden), same on every rank

# Region 2 (TENSOR-PARALLEL): column-parallel X@A then GeLU, each rank owns a slice of ffn...
H_shards = [gelu(ln_full @ Ai) for Ai in A_shards]          # rank i -> (seq, ffn/W)
# ...then row-parallel H@B: each rank produces a PARTIAL (seq, hidden) sum.
Y_partials = [Hi @ Bi for Hi, Bi in zip(H_shards, B_shards)]  # each (seq, hidden), partial sums

# g_bar = reduce_scatter(SEQ): SUM the row-parallel partials (== TP's all_reduce sum) AND scatter
#         back into SEQUENCE shards in ONE collective -> we are sequence-parallel again.
Y_seq_shards = reduce_scatter(Y_partials, axis=0)          # list of W shards, each (seq/W, hidden)
print(f"after reduce_scatter: per-rank output shape = {Y_seq_shards[0].shape} (back to seq-sharded)")

# Reconstruct the full output (all_gather the final seq-shards) for the correctness check.
Y_parallel = all_gather(Y_seq_shards, axis=0)


# ---------------------------------------------------------------------------
# SINGLE-DEVICE REFERENCE — the full, unsharded computation. The assert is the proof.
# ---------------------------------------------------------------------------
Y_reference = gelu(layernorm(X) @ A) @ B

np.testing.assert_allclose(Y_parallel, Y_reference, atol=1e-5)
print("collectives this forward: 1 all_gather (enter TP) + 1 reduce_scatter (leave TP) "
      "= same bytes as TP's 1 all_reduce")
print("✓ matches single-device reference")
