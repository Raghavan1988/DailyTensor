import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X using: (X - mean) / (std + eps)

    If 2D and axis=0, standardizes per column.
    Returns np.ndarray with dtype float64.

    Constant slices (std <= eps) map to 0 rather than relying on eps in the
    denominator, which would otherwise yield tiny-but-nonzero noise values.
    """
    X = np.asarray(X, dtype=np.float64)

    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, ddof=0, keepdims=True)

    constant = std <= eps
    safe_std = np.where(constant, 1.0, std)
    return (X - mean) / safe_std
