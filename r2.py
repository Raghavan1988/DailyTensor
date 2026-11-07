import numpy as np


def r2_score(y_true, y_pred) -> float:
    """
    Coefficient of determination: R^2 = 1 - SS_res / SS_tot.

    Returns a Python float. When y_true is constant (SS_tot == 0), R^2 is
    undefined; by convention return 1.0 for a perfect fit and 0.0 otherwise.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Mismatched shapes would broadcast silently and produce a meaningless score.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("r2_score is undefined for empty input")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)
