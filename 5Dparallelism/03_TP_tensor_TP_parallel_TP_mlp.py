"""
Tensor-Parallel Transformer MLP -- the Megatron payoff: ONE all_reduce per block.
================================================================================

CONCEPT
-------
A transformer MLP block is just:  Y = GeLU(X @ A) @ B
    X : (tokens, d_model)      A : (d_model, d_ff)      B : (d_ff, d_model)
We want to split A and B across W GPUs so no single GPU holds the full weights
*and* so we communicate as little as possible. Megatron's key insight is the
PAIRING of two shardings that cancel each other's need to gather:

    A is COLUMN-parallel : shard A along axis=1 (its OUTPUT dim, d_ff).
                           A_i is (d_model, d_ff/W).  Rank i computes
                           h_i = GeLU(X @ A_i), a (tokens, d_ff/W) column-slice.
    B is ROW-parallel    : shard B along axis=0 (its INPUT dim, d_ff).
                           B_i is (d_ff/W, d_model).  Rank i computes
                           Y_partial_i = h_i @ B_i, a (tokens, d_model) PARTIAL sum.

WHY THIS PAIRING IS MAGIC
-------------------------
1. GeLU is ELEMENTWISE. Applying it to a column-slice h_i gives exactly the same
   numbers as slicing GeLU of the full activation. So GeLU needs NO communication
   and the big intermediate activation h (tokens x d_ff) is NEVER materialized or
   gathered -- each rank only ever holds its (tokens x d_ff/W) slice.
2. The column-shard of h lines up perfectly with the row-shard of B: contracting
   h_i (d_ff/W wide) against B_i (d_ff/W tall) is one term of the full sum over
   d_ff. Summing the W partials reconstructs the true Y = h @ B.

WHICH AXIS / WHICH COLLECTIVE / WHY
-----------------------------------
  Sharded axis:  A on axis=1, B on axis=0 (the shared contraction dim d_ff).
  Collective:    all_reduce(SUM) over the per-rank Y_partial_i.  Used because
                 row-parallel B turns matmul into a sum of partials, and SUM of
                 partials == the exact dense result.
COMMUNICATION COST: exactly ONE all_reduce per forward (and one in backward).
The intermediate activation h stays sharded the ENTIRE time; we talk to peers
only ONCE, at the block boundary.  <-- this is the whole lesson.

We simulate W GPUs in ONE process; each 'rank' only ever touches its own shard.
"""

import numpy as np

W = 4  # world size: number of simulated GPUs / ranks


# ---------------------------------------------------------------------------
# Collective: the ONE communication primitive this block needs.
# ---------------------------------------------------------------------------
def all_reduce(shards):
    """Elementwise SUM of every rank's tensor; identical on all ranks (== dist.all_reduce, op=SUM)."""
    total = np.sum(shards, axis=0)          # the single cross-rank exchange
    return [total.copy() for _ in shards]   # broadcast: every rank gets the sum


def split(x, W, axis):
    """Helper: split x into W equal shards along 'axis' (the per-rank scatter of a full tensor)."""
    return np.split(x, W, axis=axis)


def gelu(x):
    """Elementwise GeLU (tanh approximation). Elementwise => safe on a column-shard."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# Problem setup (deterministic).
# ---------------------------------------------------------------------------
np.random.seed(0)
tokens, d_model, d_ff = 6, 8, 16          # d_ff must be divisible by W
X = np.random.randn(tokens, d_model)
A = np.random.randn(d_model, d_ff)        # column-parallel weight (split axis=1)
B = np.random.randn(d_ff, d_model)        # row-parallel weight    (split axis=0)

# ---- SINGLE-DEVICE REFERENCE: the full, unsharded MLP block. ----
reference = gelu(X @ A) @ B               # Y = GeLU(X @ A) @ B  -> (tokens, d_model)

# ---------------------------------------------------------------------------
# Shard the weights across ranks. X is REPLICATED (every rank holds the full X).
# ---------------------------------------------------------------------------
A_shards = split(A, W, axis=1)            # A_i : (d_model, d_ff/W)  -- column-parallel
B_shards = split(B, W, axis=0)            # B_i : (d_ff/W, d_model)  -- row-parallel
print(f"world size W={W}   X is replicated {X.shape}")
print(f"A column-parallel (axis=1): full {A.shape} -> per-rank {A_shards[0].shape}")
print(f"B row-parallel    (axis=0): full {B.shape} -> per-rank {B_shards[0].shape}\n")

# ---------------------------------------------------------------------------
# FORWARD PASS -- each rank touches only its own shard; NO comm until the end.
# ---------------------------------------------------------------------------
Y_partials = []
for i in range(W):
    # 1) Column-parallel matmul + elementwise GeLU. h_i is a COLUMN-SLICE of h
    #    and is NEVER gathered -- the wide activation stays sharded.
    h_i = gelu(X @ A_shards[i])           # (tokens, d_ff/W)
    # 2) Row-parallel matmul -> a PARTIAL Y (one term of the sum over d_ff).
    Y_partial_i = h_i @ B_shards[i]       # (tokens, d_model)
    Y_partials.append(Y_partial_i)
    print(f"rank {i}: h_i={h_i.shape} (stays sharded)  ->  Y_partial_i={Y_partial_i.shape}")

# ---------------------------------------------------------------------------
# THE ONE COMMUNICATION STEP: all_reduce(SUM) the partials at the block boundary.
# ---------------------------------------------------------------------------
ALL_REDUCE_COUNT = 0
print("\n>>> all_reduce(SUM) over the W partial Y's  (the block's ONLY collective) <<<")
Y_ranks = all_reduce(Y_partials)
ALL_REDUCE_COUNT += 1
Y = Y_ranks[0]                            # identical on every rank after all_reduce

# ---------------------------------------------------------------------------
# PROOF: the tensor-parallel result equals the single-device reference.
# ---------------------------------------------------------------------------
np.testing.assert_allclose(Y, reference, atol=1e-5)
print(f"\nall_reduce count this forward = {ALL_REDUCE_COUNT}  (exactly 1, as promised)")
print("communication cost: ONE all_reduce per MLP forward; h never gathered.")
print("✓ matches single-device reference")
