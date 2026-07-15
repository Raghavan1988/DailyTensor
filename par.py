"""
Column-Parallel Linear  —  Tensor Parallelism, Part 1 (Megatron "A" matrix)
===========================================================================

CONCEPT
    A linear layer  Y = X @ Wm (+ b),  with  Wm of shape (in, out).
    COLUMN parallelism shards Wm along its OUTPUT dimension (axis=1) across the
    P ranks (P = world size, named W in the code below):
        Wm -> [Wm_0, Wm_1, ..., Wm_{P-1}],  each Wm_i of shape (in, out / P).
    The input X is REPLICATED on every rank (every rank holds the full X).
    Each rank computes an independent vertical slice of the output:
        Y_i = X @ Wm_i       # shape (batch, out/P),  NO communication needed.

WHICH AXIS IS SHARDED
    W is split along axis=1 (the output / column axis) -> hence "column parallel".
    Y therefore comes out split along its last axis (the columns of Y).

WHICH COLLECTIVE & WHY
    To MATERIALIZE the full Y we must glue the column-slices back together,
    so the forward pass uses exactly ONE  all_gather  along the last axis.
    We gather (not reduce) because each rank owns *different* output columns
    that are concatenated, not summed.

COMMUNICATION COST
    Forward = 1 all_gather (over the last axis). That's it.
    NOTE (sets up program 03): if the very next layer is a ROW-parallel linear,
    it WANTS its input already split along this same axis — so we never gather
    here at all; the column-sharded activation flows straight into the row layer,
    whose final all_reduce is then the block's only collective.

We simulate W GPUs in ONE process; each "rank" only ever touches its own shard.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Collective communication primitives (tiny, explicit, named to match NCCL).
# Only all_gather is needed for column-parallel forward.
# ----------------------------------------------------------------------------
def all_gather(shards, axis):
    """Concatenate every rank's shard along `axis`; identical on all ranks (== dist.all_gather)."""
    return np.concatenate(shards, axis=axis)


def split(x, W, axis):
    """Helper: split x into W equal shards along `axis` (the inverse of all_gather)."""
    return np.split(x, W, axis=axis)


# ----------------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------------
np.random.seed(0)

W = 4              # world size: we pretend there are 4 GPUs
batch = 3
in_dim = 8
out_dim = 12       # must be divisible by W so columns split evenly

X = np.random.randn(batch, in_dim)        # input, REPLICATED on every rank
Wm = np.random.randn(in_dim, out_dim)     # full weight matrix  (in, out)
b = np.random.randn(out_dim)              # full bias, indexed per column-slice

# ----------------------------------------------------------------------------
# SINGLE-DEVICE REFERENCE: the full, unsharded computation we must reproduce.
# ----------------------------------------------------------------------------
reference = X @ Wm + b                     # shape (batch, out_dim)

# ----------------------------------------------------------------------------
# SHARD the weight along axis=1 (the OUTPUT/column dimension).
# Each rank receives one vertical slice W_i of shape (in, out/W).
# The bias is split the same way, since each column of Y has its own bias.
# ----------------------------------------------------------------------------
W_shards = split(Wm, W, axis=1)            # list of (in, out/W) matrices
b_shards = split(b, W, axis=0)             # list of (out/W,) vectors

print(f"world size W = {W}")
print(f"full   W shape = {Wm.shape}   (in={in_dim}, out={out_dim})")
print(f"per-rank W_i shape = {W_shards[0].shape}   (in, out/W = {out_dim // W})")
print(f"X is REPLICATED on every rank, shape = {X.shape}\n")

# ----------------------------------------------------------------------------
# PER-RANK COMPUTE: each rank does its own matmul on its own columns.
# This loop is the "parallel" region — every iteration is an independent GPU.
# There is NO communication inside this loop: pure local matmul. <-- the point
# ----------------------------------------------------------------------------
Y_shards = []
for rank in range(W):
    Y_i = X @ W_shards[rank] + b_shards[rank]   # (batch, out/W), no comms
    Y_shards.append(Y_i)
    print(f"rank {rank}: Y_i = X @ W_{rank} -> shape {Y_i.shape}  (owns output cols "
          f"{rank * (out_dim // W)}..{(rank + 1) * (out_dim // W) - 1})")

# ----------------------------------------------------------------------------
# THE LESSON: one all_gather along the LAST axis stitches the column-slices
# into the full Y. This is the ONLY collective in the forward pass.
# ----------------------------------------------------------------------------
print(f"\n>>> COLLECTIVE: all_gather over axis=-1 (concatenating {W} column-slices)")
parallel = all_gather(Y_shards, axis=-1)        # (batch, out_dim)
print(f"    gathered Y shape = {parallel.shape}  (== full reference shape)\n")

# ----------------------------------------------------------------------------
# PROOF: the gathered parallel result equals the single-device computation.
# ----------------------------------------------------------------------------
np.testing.assert_allclose(parallel, reference, atol=1e-5)
print("✓ matches single-device reference")
