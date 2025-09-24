
import numpy as np

def expanding_windows(n, initial, step=1):
    """Yield (train_idx, test_idx) tuples for expanding-window walk-forward."""
    start = initial
    while start + step <= n:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, start + step)
        yield train_idx, test_idx
        start += step
