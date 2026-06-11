import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X using: (X - mean) / (std + eps)

    If 2D and axis=0, standardizes per column.
    Returns np.ndarray with dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)

    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, ddof=0, keepdims=True)

    standardized = (X - mean) / (std + eps)

    return standardized.astype(np.float64)
