import numpy as np


def expected_value(x, p):
    """
    Expected value of a discrete random variable: E[X] = sum_i x_i * p_i.

    `p` must be a valid probability distribution: non-negative and summing to 1.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    # Negative weights can still sum to 1, so check them separately.
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1")

    return float(np.sum(x * p))


def expected_value_discrete(x, p):
    """Alias of expected_value kept for backward compatibility."""
    return expected_value(x, p)
