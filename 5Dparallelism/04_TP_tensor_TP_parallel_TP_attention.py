"""
Tensor-Parallel Multi-Head Attention — heads sharded across ranks.
=================================================================

CONCEPT
-------
Multi-Head Attention runs `n_heads` INDEPENDENT attention heads in parallel and
concatenates their outputs.  Because the heads are independent, attention shards
along the HEAD axis: with W ranks, each rank owns `n_heads // W` heads and does
all of that head's math locally.

WHICH AXIS IS SHARDED
---------------------
The HEAD axis.  Concretely:
  * Fused QKV projection  W_qkv  (COLUMN-parallel):
        output dim = n_heads * head_dim, split BY HEADS down the columns.
        Each rank projects the full X into Q_i, K_i, V_i for ONLY its heads.
  * Per-head attention  softmax(Q K^T / sqrt(d)) V  is FULLY LOCAL.
        Heads never talk to each other, so there is NO communication here.
  * Output projection  W_o  (ROW-parallel):
        its input dim is the concatenated head outputs, split BY HEADS down the
        rows.  Each rank multiplies its local context by its slice W_o_i to get
        a PARTIAL output (full d_model shape, but only its heads contributed).

WHICH COLLECTIVE & WHY
----------------------
The W partial outputs must be SUMMED to reconstruct  concat(heads) @ W_o, because
  concat(ctx_0..ctx_{W-1}) @ [W_o_0; ...; W_o_{W-1}]  ==  sum_i ctx_i @ W_o_i.
That sum is exactly an ALL_REDUCE (op=SUM): one collective, result identical on
every rank.

COMMUNICATION COST
------------------
Exactly ONE all_reduce per attention forward pass (same as the MLP block).  The
column-parallel QKV and the local softmax need zero communication; only the
row-parallel output projection's partial sums must be combined.

We simulate W GPUs in ONE process; each 'rank' only ever touches its own shard.
"""

import numpy as np

np.random.seed(0)

# -- Collectives -------------------------------------------------------------
# Tiny explicit stand-ins for torch.distributed / NCCL ops.  In a real cluster
# these move bytes between GPUs; here we just operate on a Python list of shards.

def all_reduce(shards):
    """Elementwise SUM of every rank's tensor; identical on all ranks (== dist.all_reduce, op=SUM)."""
    total = np.sum(shards, axis=0)          # the single cross-rank exchange
    return [total.copy() for _ in shards]   # broadcast: every rank gets its own copy of the sum


def split(x, W, axis):
    """Helper: split x into W equal shards along 'axis' (the per-rank scatter of a full tensor)."""
    return np.split(x, W, axis=axis)

# -- Problem setup -----------------------------------------------------------
W = 2            # world size (ranks / GPUs); each rank gets n_heads // W heads
n_heads = 4      # total attention heads
head_dim = 8     # dimension per head
d_model = n_heads * head_dim   # 32 — model/embedding width
batch, seq = 2, 3
heads_per_rank = n_heads // W

X = np.random.randn(batch, seq, d_model)            # shared input, replicated on every rank
W_qkv = np.random.randn(3, d_model, n_heads, head_dim)  # fused Q,K,V projection (per head)
W_o = np.random.randn(n_heads, head_dim, d_model)       # output projection (per head, row-parallel)


def attention_head(q, k, v):
    """Single-head scaled-dot-product attention with a numerically stable softmax."""
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)   # (batch, seq, seq)
    scores -= scores.max(axis=-1, keepdims=True)            # stability: subtract row max
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v                                            # (batch, seq, head_dim)

# -- Single-device reference (full, unsharded computation) -------------------
def reference():
    """The whole attention block on one device — the ground truth."""
    out = np.zeros((batch, seq, d_model))
    for h in range(n_heads):
        q = X @ W_qkv[0, :, h]      # (batch, seq, head_dim)
        k = X @ W_qkv[1, :, h]
        v = X @ W_qkv[2, :, h]
        ctx = attention_head(q, k, v)
        out += ctx @ W_o[h]         # accumulate each head's contribution
    return out

# -- Tensor-parallel computation (heads sharded across W ranks) ---------------
def tensor_parallel():
    head_groups = split(np.arange(n_heads), W, axis=0)   # which heads each rank owns
    partials = []                                        # one partial output per rank
    for rank in range(W):
        my_heads = head_groups[rank]
        # COLUMN-parallel QKV: project X into Q/K/V for ONLY this rank's heads.
        ctx_local = []
        for h in my_heads:
            q = X @ W_qkv[0, :, h]
            k = X @ W_qkv[1, :, h]
            v = X @ W_qkv[2, :, h]
            print(f"  rank {rank} head {h}: Q/K/V shape {q.shape}")
            ctx_local.append(attention_head(q, k, v))    # LOCAL softmax — no comms
        # ROW-parallel W_o: partial output = sum over this rank's heads of ctx @ W_o_h.
        partial = np.zeros((batch, seq, d_model))
        for h, ctx in zip(my_heads, ctx_local):
            partial += ctx @ W_o[h]
        partials.append(partial)
        print(f"rank {rank} owns heads {list(my_heads)} -> partial output {partial.shape}")
    # THE LESSON: combine the W row-parallel partial sums with ONE all_reduce.
    # (The head axis was already contracted away by `ctx @ W_o[h]`, so this is a
    #  plain elementwise sum of full (batch, seq, d_model) partials — no head axis.)
    print(f"all_reduce: elementwise SUM of {W} full (batch,seq,d_model) partials -> full output  [1 collective]")
    return all_reduce(partials)[0]


if __name__ == "__main__":
    print(f"world W={W}, n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}, "
          f"heads/rank={heads_per_rank}\n")
    parallel = tensor_parallel()
    np.testing.assert_allclose(parallel, reference(), atol=1e-5)
    print("\n✓ matches single-device reference")
