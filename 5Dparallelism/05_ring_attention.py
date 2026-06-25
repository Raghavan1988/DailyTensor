"""
Ring Attention / Context Parallelism, NumPy simulation.

Tensor parallelism splits the hidden dimension; this is the other axis --
splitting the SEQUENCE so we can attend over contexts too long to fit (in
activation memory) on a single device.  The sequence is cut into P blocks
placed on P devices arranged in a ring:

    device i permanently holds its query block Q_i, and the K/V blocks are
    passed around the ring.  After P steps every device has seen every K/V
    block, so it has computed full attention for its queries -- yet no device
    ever stored more than one K/V block at a time.

The enabling trick is ONLINE SOFTMAX (the same idea as FlashAttention): the
softmax over the whole key sequence can be accumulated one K/V block at a time
by tracking a running row-max `m`, a running denominator `l`, and a running
weighted output `acc`, rescaling as each block arrives.  Because this
accumulation is associative, the block ORDER doesn't matter -- which is exactly
why rotating K/V around a ring is allowed.

We verify the streamed, blockwise result matches vanilla full attention.
(For decoder LMs you'd add a causal mask + a load-balanced ring schedule; this
demo uses full bidirectional attention to keep the core idea uncluttered.)
"""

import numpy as np


def softmax_attention(Q, K, V):
    """Reference: standard full attention with every key visible at once."""
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(-1, keepdims=True)) @ V


def ring_attention(Q, K_blocks, V_blocks):
    """
    Attention for one query block Q, streaming over the K/V blocks in the order
    they arrive around the ring, using online softmax.  Mathematically identical
    to softmax_attention(Q, concat(K_blocks), concat(V_blocks)).
    """
    n, d = Q.shape
    m = np.full((n, 1), -np.inf)     # running row-max of the scores seen so far
    l = np.zeros((n, 1))             # running sum of exp(score - m)
    acc = np.zeros((n, d))           # running un-normalised output

    for K_blk, V_blk in zip(K_blocks, V_blocks):     # one ring step per block
        s = Q @ K_blk.T / np.sqrt(d)                 # [n, block] block scores
        m_new = np.maximum(m, s.max(-1, keepdims=True))
        alpha = np.exp(m - m_new)                    # rescale factor for old state
        p = np.exp(s - m_new)                        # softmax weights for this block
        l = alpha * l + p.sum(-1, keepdims=True)     # update denominator
        acc = alpha * acc + p @ V_blk                # update weighted output
        m = m_new
    return acc / l                                   # final normalisation


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq, d, P = 12, 8, 4
    assert seq % P == 0
    Q = rng.standard_normal((seq, d))
    K = rng.standard_normal((seq, d))
    V = rng.standard_normal((seq, d))

    reference = softmax_attention(Q, K, V)

    # split K and V into P ring blocks along the sequence dimension
    K_blocks = np.split(K, P, axis=0)
    V_blocks = np.split(V, P, axis=0)

    # each device owns one query block and streams all K/V blocks past it
    Q_blocks = np.split(Q, P, axis=0)
    out_blocks = [ring_attention(Qi, K_blocks, V_blocks) for Qi in Q_blocks]
    out = np.concatenate(out_blocks, axis=0)

    print("max abs diff vs full attention             :", np.abs(out - reference).max())
    print("online-softmax over blocks == full softmax :", np.allclose(out, reference))
    print("\nEach device held seq/P =", seq // P,
          "queries and one K/V block at a time, never all", seq, "keys.")
