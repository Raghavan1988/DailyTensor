import numpy as np


def cosine_similarity(a, b, eps=1e-12):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0

    sim = np.dot(a, b) / (norm_a * norm_b)
    # Clip away floating-point drift just outside [-1, 1].
    return float(np.clip(sim, -1.0, 1.0))
