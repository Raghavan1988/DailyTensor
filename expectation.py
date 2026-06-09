import numpy as np

def expected_value(x, p):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1")

    return float(np.sum(x * p))

    
def expected_value_discrete(x, p):
    return expected_value(x,p)
