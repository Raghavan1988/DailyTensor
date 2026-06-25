"""
Row-parallel Linear  —  Tensor Parallelism, part 2 (Megatron "B" matrix).
============================================================================

CONCEPT
-------
A linear layer computes  Y = X @ W + b,  with W of shape (in, out).
ROW parallelism shards W along its INPUT dimension (axis=0):

        W = [ W_0 ]          each W_i has shape (in/W, out)
            [ W_1 ]
            [ ... ]
            [ W_{W-1} ]

Because the contraction is over the input dim, the INPUT X must ALSO be
sharded along its last axis (axis=1) so that X_i lines up with W_i:

        X = [ X_0 | X_1 | ... | X_{W-1} ]    each X_i has shape (batch, in/W)

Each rank then computes a PARTIAL product:

        partial_i = X_i @ W_i        shape (batch, out)  -- FULL output shape!

Note partial_i already has the full output shape, but it is only a PARTIAL
sum: it accounts for just this rank's slice of the input dimension. The true
result is the sum over all ranks,  Y = sum_i (X_i @ W_i),  which is exactly
the block-wise rule for matrix multiplication.

WHICH AXIS IS SHARDED:  W on axis=0 (input dim); X on axis=1 (its last dim).
WHICH COLLECTIVE & WHY: ALL_REDUCE (SUM). Every rank holds a different partial
                        of the SAME shape; summing them reconstructs Y and
                        leaves the identical full result on every rank.
COMMUNICATION COST:     exactly 1 all_reduce per forward pass.

The bias b is NOT sharded. Adding it on every rank before the reduce would add
it W times, so we add b ONCE, after the all_reduce.

WHY THIS PAIRS WITH PROGRAM 01 (column-parallel)
------------------------------------------------
A column-parallel layer (program 01) leaves its output sharded along the
output/feature axis. A row-parallel layer WANTS its input sharded along that
exact axis. So column-parallel -> row-parallel composes with NO gather in
between: the sharded activations flow straight through, and only one
all_reduce (at the end of the row layer) is needed. That fused pattern is the
classic Megatron MLP block, and it is the setup for program 03.

SIMULATION NOTE
---------------
We simulate W GPUs in ONE process; each 'rank' only ever touches its own shard.
There are no real processes, threads, or message passing -- the "collective"
is just a Python function over a list of shards, which is precisely what the
NCCL/torch.distributed collective does across real devices.
"""

import numpy as np

np.random.seed(0)

W = 4  # world size: number of simulated GPUs / ranks


# ---------------------------------------------------------------------------
# Collectives — tiny explicit stand-ins for torch.distributed / NCCL ops.
# Only all_reduce is needed for a row-parallel forward; split is a setup helper.
# ---------------------------------------------------------------------------
def all_reduce(shards):
    """Elementwise SUM of every rank's tensor; identical on all ranks (== dist.all_reduce, op=SUM)."""
    total = np.sum(shards, axis=0)          # the single cross-rank exchange
    return [total.copy() for _ in shards]   # broadcast: every rank gets its own copy of the sum


def split(x, W, axis):
    """Split x into W equal shards along 'axis' (the per-rank scatter of a full tensor)."""
    return list(np.split(x, W, axis=axis))


# ---------------------------------------------------------------------------
# Problem setup: full (unsharded) tensors live here only to BUILD the shards
# and to serve as the single-device reference. No rank ever reads them whole.
# ---------------------------------------------------------------------------
batch, in_dim, out_dim = 3, 8, 5
X = np.random.randn(batch, in_dim)        # (batch, in)
Wmat = np.random.randn(in_dim, out_dim)   # (in, out)
b = np.random.randn(out_dim)              # (out,)  -- NOT sharded

# Shard the weight along its INPUT dim (axis=0) -> W rows-blocks.
W_shards = split(Wmat, W, axis=0)         # each (in/W, out)
# Shard the input along its LAST dim (axis=1) so X_i pairs with W_i.
X_shards = split(X, W, axis=1)            # each (batch, in/W)

print(f"World size W = {W}")
print(f"Full   X    shape {X.shape}, full W shape {Wmat.shape}, full b shape {b.shape}")
print(f"Per-rank W_i shape {W_shards[0].shape}  (in/W, out)  <- sharded on axis=0 (input dim)")
print(f"Per-rank X_i shape {X_shards[0].shape}  (batch, in/W) <- sharded on axis=1 (last dim)")


# ---------------------------------------------------------------------------
# Parallel forward: each rank computes its PARTIAL, then ALL_REDUCE sums them.
# ---------------------------------------------------------------------------
partials = []
for rank in range(W):
    partial = X_shards[rank] @ W_shards[rank]   # (batch, out) -- full shape, partial value
    print(f"  rank {rank}: X_{rank}{X_shards[rank].shape} @ W_{rank}{W_shards[rank].shape}"
          f" -> partial{partial.shape}  (a PARTIAL sum)")
    partials.append(partial)

print(f"--> all_reduce SUM over {W} partials of shape {partials[0].shape}"
      f" (each rank ends up with the identical full result)")
summed = all_reduce(partials)[0]   # THE LESSON: 1 collective reconstructs Y (take this rank's copy)
Y_parallel = summed + b            # add the (unsharded) bias ONCE, after the reduce


# ---------------------------------------------------------------------------
# Single-device reference + proof.
# ---------------------------------------------------------------------------
Y_reference = X @ Wmat + b

print(f"Reduced+bias output shape {Y_parallel.shape} == reference {Y_reference.shape}")
np.testing.assert_allclose(Y_parallel, Y_reference, atol=1e-5)
print("✓ matches single-device reference")
