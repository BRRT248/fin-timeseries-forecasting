
from src.utils.metrics import sharpe_ratio, max_drawdown, hit_rate
import numpy as np

def test_metrics_basic():
    r = np.array([0.01, -0.005, 0.002])
    s = sharpe_ratio(r)
    h = hit_rate(r)
    ec = (1+r).cumprod()
    mdd = max_drawdown(ec)
    assert np.isfinite(s)
    assert 0 <= h <= 1
    assert mdd <= 0
