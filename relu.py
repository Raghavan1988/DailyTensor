import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0)
