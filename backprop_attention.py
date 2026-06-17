#### Educational: Backpropagation through a Self-Attention block
"""
Forward + MANUAL backward pass through a single-head self-attention layer.
The point of this file is to make every gradient — especially the ones
flowing back through the Q / K / V projections — completely explicit so
you can see exactly how the chain rule applies.

Forward graph (no batch axis; n = sequence length):

    Q = X @ Wq                  # (n, dk)        queries
    K = X @ Wk                  # (n, dk)        keys
    V = X @ Wv                  # (n, dv)        values
    S = (Q @ K^T) * scale       # (n, n)         scaled scores
    A = softmax(S, axis=-1)     # (n, n)         attention weights (rows sum to 1)
    O = A @ V                   # (n, dv)        attention output

We assume the downstream loss L is a scalar function of O. PyTorch /
autograd would compute dL/dO for us; here we receive it as `dO` and walk
the graph in reverse, computing the local gradient at each step and
matmul'ing it into the upstream gradient.

Two cheat-sheet rules used over and over below:

  Matmul rule:   if  Z = U @ W   then   dU = dZ @ W^T   and   dW = U^T @ dZ
  Sum rule:      if a tensor feeds into MULTIPLE downstream branches, its
                 gradient is the SUM of the gradients flowing back from
                 each branch. (X is used three times — for Q, K, AND V.)

At the bottom we verify every analytic gradient against a centered
finite-difference estimate, which should match to ~1e-7.
"""

import numpy as np


def softmax(x, axis=-1):
    # Numerically stable: subtract the row max before exp() so we don't overflow.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------
def attention_forward(X, Wq, Wk, Wv):
    """
    Inputs:
        X:  (n, d_model)   token embeddings going into the attention block
        Wq: (d_model, dk)  query projection weights
        Wk: (d_model, dk)  key   projection weights
        Wv: (d_model, dv)  value projection weights

    Returns:
        O:     (n, dv)  attention output
        cache: dict of intermediates the backward pass will need
    """
    # ---- Linear projections that produce Queries, Keys, and Values ----
    # Each row of X is one token's embedding. Multiplying on the right by
    # Wq projects every token into the "query" subspace; same for K and V.
    # These three linear layers are where Wq/Wk/Wv learn what aspect of the
    # input to compare (Q vs K) and what to mix together (V).
    Q = X @ Wq                              # (n, dk)
    K = X @ Wk                              # (n, dk)
    V = X @ Wv                              # (n, dv)

    # ---- Scaled dot-product scores -----------------------------------
    # S[i, j] is the unnormalized "affinity" between query i and key j.
    # Without the 1/sqrt(dk) scale, the dot products grow with dk and push
    # the softmax into saturated regions where gradients vanish.
    dk = Q.shape[-1]
    scale = 1.0 / np.sqrt(dk)
    S = (Q @ K.T) * scale                   # (n, n)

    # ---- Softmax along the KEYS axis ---------------------------------
    # Row i of A is a probability distribution: how much query i attends
    # to each key j. Rows are independent — softmax is applied per row.
    A = softmax(S, axis=-1)                 # (n, n)

    # ---- Weighted sum of value vectors -------------------------------
    # The output for query i is the convex combination of value vectors
    # weighted by attention weights A[i, :]. This is the "what to mix"
    # half of attention — V supplies the content, A decides how much.
    O = A @ V                               # (n, dv)

    # Cache everything backprop needs. We must save A and V (used by the
    # softmax and the A@V backward), X (used by every weight gradient),
    # and the weight matrices themselves (used to compute dX).
    cache = dict(X=X, Wq=Wq, Wk=Wk, Wv=Wv,
                 Q=Q, K=K, V=V, A=A, scale=scale)
    return O, cache


# ----------------------------------------------------------------------
# Backward pass — the interesting bit
# ----------------------------------------------------------------------
def attention_backward(dO, cache):
    """
    Input:
        dO:    (n, dv)  gradient of the loss w.r.t. the attention output O
        cache: dict produced by attention_forward

    Returns gradients of the loss w.r.t. every input/parameter:
        dX, dWq, dWk, dWv
    """
    X, Wq, Wk, Wv = cache["X"], cache["Wq"], cache["Wk"], cache["Wv"]
    Q, K, V, A, scale = cache["Q"], cache["K"], cache["V"], cache["A"], cache["scale"]

    # =================================================================
    # Step 1: backprop through   O = A @ V
    # -----------------------------------------------------------------
    # Apply the matmul rule with U=A, W=V, Z=O:
    #     dA = dO @ V^T       <- upstream gradient routed through V's transpose
    #     dV = A^T @ dO       <- upstream gradient routed through A's transpose
    # Intuition: each output row O[i] is a linear combination of value
    # rows weighted by A[i, :]. So the gradient flowing INTO A[i, j]
    # equals dO[i] · V[j], which is exactly (dO @ V^T)[i, j].
    # =================================================================
    dA = dO @ V.T                            # (n, n)
    dV = A.T @ dO                            # (n, dv)

    # =================================================================
    # Step 2: backprop through   A = softmax(S, axis=-1)
    # -----------------------------------------------------------------
    # Softmax is applied row-wise, so each row is its own little function.
    # For a single row a = softmax(s), the Jacobian is
    #     da_i / ds_j = a_i * (delta_ij - a_j)
    # which gives the well-known closed form
    #     dL/ds_j = a_j * ( dL/da_j  -  sum_k a_k * dL/da_k )
    # Vectorized across all rows at once:
    #     dS = A * ( dA  -  sum(A * dA, axis=-1, keepdims=True) )
    # We never have to materialize the full (n, n, n) Jacobian.
    # =================================================================
    row_dot = np.sum(A * dA, axis=-1, keepdims=True)   # (n, 1)
    dS = A * (dA - row_dot)                            # (n, n)

    # =================================================================
    # Step 3: backprop through   S = (Q @ K^T) * scale
    # -----------------------------------------------------------------
    # First push the upstream gradient through the constant scalar `scale`
    # (the derivative of (Q @ K^T) * scale w.r.t. (Q @ K^T) is just scale).
    # Then apply the matmul rule, but watch out: the second factor here
    # is K^T, not K, so the gradients pick up the corresponding transposes.
    #
    #     For  Z = Q @ K^T       (i.e. Z_ij = sum_l Q[i,l] * K[j,l]):
    #         dQ[i,l] = sum_j dZ[i,j] * K[j,l]    -> dQ = dZ @ K
    #         dK[j,l] = sum_i dZ[i,j] * Q[i,l]    -> dK = dZ^T @ Q
    #
    # This is where the symmetry between Q and K becomes obvious: they
    # play structurally identical roles in the score matrix, and their
    # gradients are just transposes of each other.
    # =================================================================
    dS_scaled = dS * scale                   # (n, n)
    dQ = dS_scaled @ K                       # (n, dk)
    dK = dS_scaled.T @ Q                     # (n, dk)

    # =================================================================
    # Step 4: backprop through the three linear projections
    #            Q = X @ Wq,   K = X @ Wk,   V = X @ Wv
    # -----------------------------------------------------------------
    # Each projection is a vanilla matmul, so the rule from Step 1 reapplies:
    #     dWq = X^T @ dQ      dWk = X^T @ dK      dWv = X^T @ dV
    # The weight gradients are straightforward.
    #
    # X is more subtle: it feeds into Q, K, AND V. By the SUM rule, its
    # gradient is the sum of the three branches' contributions. Each
    # branch contributes (upstream_grad @ W^T):
    #     dX = dQ @ Wq^T  +  dK @ Wk^T  +  dV @ Wv^T
    # If you forget one of these terms you'll silently train a broken
    # model whose embeddings only update through some of the heads.
    # =================================================================
    dWq = X.T @ dQ                           # (d_model, dk)
    dWk = X.T @ dK                           # (d_model, dk)
    dWv = X.T @ dV                           # (d_model, dv)

    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T   # (n, d_model)

    return dX, dWq, dWk, dWv


# ----------------------------------------------------------------------
# Self-test: verify every analytic gradient against finite differences.
# ----------------------------------------------------------------------
def _numerical_grad(f, x, eps=1e-6):
    """Centered finite-difference gradient of scalar f() w.r.t. array x (mutated in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; f_plus  = f()
        x[idx] = old - eps; f_minus = f()
        x[idx] = old                                 # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


if __name__ == "__main__":
    np.random.seed(0)
    n, d_model, dk, dv = 4, 6, 5, 3
    X  = np.random.randn(n, d_model)
    Wq = np.random.randn(d_model, dk)
    Wk = np.random.randn(d_model, dk)
    Wv = np.random.randn(d_model, dv)

    # Define a synthetic scalar loss L = sum(O * dO_target). With this choice
    # dL/dO is EXACTLY dO_target, so we can hand-pick the upstream gradient
    # that flows into the backward pass.
    dO_target = np.random.randn(n, dv)

    def loss():
        O, _ = attention_forward(X, Wq, Wk, Wv)
        return float(np.sum(O * dO_target))

    # Analytic gradients via our hand-coded backward pass.
    O, cache = attention_forward(X, Wq, Wk, Wv)
    dX, dWq, dWk, dWv = attention_backward(dO_target, cache)

    # Compare each one against a numerical gradient. Any disagreement
    # larger than ~1e-6 would indicate a bug in the analytic formulas.
    print("Verifying analytic gradients against numerical gradients:")
    for name, param, analytic in [("X",  X,  dX),
                                  ("Wq", Wq, dWq),
                                  ("Wk", Wk, dWk),
                                  ("Wv", Wv, dWv)]:
        numeric = _numerical_grad(loss, param)
        err = float(np.max(np.abs(analytic - numeric)))
        print(f"  max |d{name}_analytic - d{name}_numeric| = {err:.2e}")
