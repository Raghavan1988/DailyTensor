### https://www.tensortonic.com/problems/sigmoid-numpy
"""Four ways to implement sigmoid(x) = 1 / (1 + exp(-x))."""

import math


def sigmoid_scalar(x: float) -> float:
    """Plain scalar sigmoid using math.exp."""
    if not isinstance(x, (int, float)):
        raise TypeError(f"expected int or float, got {type(x).__name__}")
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_numpy(x):
    """Vectorized, overflow-safe sigmoid for NumPy scalars/arrays.

    Uses the identity sigmoid(x) = exp(x)/(1+exp(x)) for x<0 so we only ever
    compute exp of a non-positive number, avoiding overflow at large |x|.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    z = np.empty_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    z[~pos] = ex / (1.0 + ex)
    return z


def sigmoid_from_scratch(x: float, terms: int = 50) -> float:
    """Sigmoid without importing math — exp computed via Taylor series.

    Accurate for roughly |x| < 20; diverges for very large |x|.
    """
    # Reduce magnitude so the Taylor series converges quickly.
    abs_x = -x if x < 0 else x
    term = 1.0
    exp_abs_x = 1.0
    for n in range(1, terms + 1):
        term *= abs_x / n
        exp_abs_x += term
    # For x >= 0: σ(x) = 1/(1+exp(-x)) = 1/(1+1/exp(|x|))
    # For x <  0: σ(x) = exp(x)/(1+exp(x)) = 1/(1+exp(|x|))
    if x >= 0:
        return 1.0 / (1.0 + 1.0 / exp_abs_x)
    return 1.0 / (1.0 + exp_abs_x)


def sigmoid_torch(x):
    """PyTorch sigmoid — supports autograd."""
    import torch

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sigmoid(x)


def demo_gradient() -> None:
    """Show that the torch version flows gradients."""
    import torch

    x = torch.tensor(0.5, requires_grad=True)
    y = sigmoid_torch(x)
    y.backward()
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    print(f"  sigmoid(0.5) = {y.item():.6f}")
    print(f"  grad        = {x.grad.item():.6f}  (expected {y.item() * (1 - y.item()):.6f})")


if __name__ == "__main__":
    test_values = [-1000.0, -2.0, 0.0, 2.0, 1000.0]

    print(f"{'x':>10} | {'scalar':>12} | {'numpy':>12} | {'scratch':>12} | {'torch':>12}")
    print("-" * 72)
    for x in test_values:
        # scalar overflows at x = -1000 (exp(1000) is inf)
        try:
            s = f"{sigmoid_scalar(x):.6f}"
        except OverflowError:
            s = "OVERFLOW"

        n = f"{float(sigmoid_numpy(x)):.6f}"

        try:
            f = f"{sigmoid_from_scratch(x):.6f}"
        except OverflowError:
            f = "OVERFLOW"

        try:
            t = f"{float(sigmoid_torch(x)):.6f}"
        except Exception as e:
            t = f"err: {e.__class__.__name__}"

        print(f"{x:>10.2f} | {s:>12} | {n:>12} | {f:>12} | {t:>12}")

    print("\nGradient demo (torch):")
    try:
        demo_gradient()
    except ImportError:
        print("  (torch not installed — skipping)")
