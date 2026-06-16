import numpy as np


def l2_normalize(x):
    # Convert input to a float64 array so the division produces floats.
    x = np.asarray(x, dtype=np.float64)
    # Compute L2 norm along the last axis; keepdims=True lets it broadcast for division.
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # Replace any zero norms with 1 to avoid division by zero (we'll zero those rows out below).
    safe_norms = np.where(norms == 0, 1.0, norms)
    # Divide every element by its row's norm (or 1 if the row was all zeros).
    return x / safe_norms
