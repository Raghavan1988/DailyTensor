import numpy as np


def l2_normalize(x, axis=-1, eps=1e-12):
    """
    Scale vectors along `axis` to unit L2 norm: x / ||x||_2.

    Vectors whose norm is below `eps` (e.g. all-zero rows) are returned
    unchanged rather than divided by ~0, which would blow up to inf/NaN.

    Example:
        >>> l2_normalize([[3, 4], [0, 0]])
        array([[0.6, 0.8],
               [0. , 0. ]])
    """
    # Convert input to a float64 array so the division produces floats.
    x = np.asarray(x, dtype=np.float64)
    # Compute L2 norm along `axis`; keepdims=True lets it broadcast for division.
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    # Replace near-zero norms with 1 so those vectors pass through unchanged.
    safe_norms = np.where(norms < eps, 1.0, norms)
    return x / safe_norms
