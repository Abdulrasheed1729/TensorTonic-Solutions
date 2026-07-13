import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    return 1.0 /  (1.0 + np.exp(-1 * np.asarray(x, dtype=float)))