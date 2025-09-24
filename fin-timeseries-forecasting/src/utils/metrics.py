
import numpy as np

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    r = np.asarray(returns)
    if r.size == 0:
        return np.nan
    excess = r - risk_free / periods_per_year
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    return np.nan if sigma == 0 else mu / sigma * np.sqrt(periods_per_year)

def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    return dd.min()

def hit_rate(returns):
    r = np.asarray(returns)
    return np.mean(r > 0) if r.size else np.nan
