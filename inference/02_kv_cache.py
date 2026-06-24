"""
IDEA 2: KV cache (the key optimization in LLM serving)
======================================================
A transformer attends from each token to ALL previous tokens. Naively, to
generate token N we recompute attention over the whole length-N prefix every
step -> O(N^2) total work. But the "key" and "value" vectors for tokens we've
already processed never change. So we CACHE them and only compute the new
token's K/V each step -> O(N) total work.

This file implements a toy single-head attention, once WITHOUT a cache and once
WITH, and verifies they produce identical results while the cache does far less
work. Run:  python 02_kv_cache.py
"""

import numpy as np

D = 8  # hidden / embedding dimension


def attention(q, K, V):
    """Standard scaled dot-product attention for ONE query vector q against
    matrices K (keys) and V (values), each row a previous token."""
    scores = K @ q / np.sqrt(D)            # similarity of q to every key
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()               # softmax -> attention weights
    return weights @ V                     # weighted sum of value vectors


def project(token_vec, W):
    """Fake linear projection used to derive q/k/v from a token embedding."""
    return token_vec @ W


def run_without_cache(tokens, Wq, Wk, Wv):
    """Recompute K and V for the ENTIRE prefix at every step (wasteful)."""
    outputs, flops = [], 0
    for i in range(len(tokens)):
        prefix = tokens[: i + 1]
        K = np.array([project(t, Wk) for t in prefix])  # recomputed every step!
        V = np.array([project(t, Wv) for t in prefix])
        q = project(tokens[i], Wq)
        outputs.append(attention(q, K, V))
        flops += len(prefix)               # count K/V rows computed this step
    return np.array(outputs), flops


def run_with_cache(tokens, Wq, Wk, Wv):
    """Append each new token's K/V to a growing cache; compute each only once."""
    K_cache, V_cache, outputs, flops = [], [], [], 0
    for i in range(len(tokens)):
        K_cache.append(project(tokens[i], Wk))   # compute ONLY the new token
        V_cache.append(project(tokens[i], Wv))
        flops += 1
        q = project(tokens[i], Wq)
        outputs.append(attention(q, np.array(K_cache), np.array(V_cache)))
    return np.array(outputs), flops


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 6
    tokens = rng.normal(size=(n, D))           # fake token embeddings
    Wq, Wk, Wv = (rng.normal(size=(D, D)) for _ in range(3))

    out_no, flops_no = run_without_cache(tokens, Wq, Wk, Wv)
    out_yes, flops_yes = run_with_cache(tokens, Wq, Wk, Wv)

    print("identical outputs? ", np.allclose(out_no, out_yes))
    print(f"K/V rows computed without cache: {flops_no}  (grows as O(N^2))")
    print(f"K/V rows computed with    cache: {flops_yes}  (grows as O(N))")
