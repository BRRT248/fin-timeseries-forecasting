
import numpy as np
import pandas as pd

def load_prices(ticker="SPY", start="2015-01-01", end="2024-12-31"):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise RuntimeError("Empty download")
        px = df["Adj Close"].rename("price").to_frame()
        px.index = pd.to_datetime(px.index)
        return px
    except Exception:
        # Fallback: synthetic geometric Brownian motion
        rng = np.random.default_rng(42)
        dates = pd.date_range(start, end, freq="B")
        n = len(dates)
        mu, sigma = 0.08, 0.2  # annualized
        dt = 1/252
        shocks = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), size=n)
        log_px = np.cumsum(shocks) + np.log(100.0)
        px = np.exp(log_px)
        return pd.DataFrame({"price": px}, index=dates)

def to_returns(prices_df):
    r = np.log(prices_df["price"]).diff().rename("ret").dropna()
    return r.to_frame()

def make_features(returns, lags=5):
    df = returns.copy()
    for i in range(1, lags+1):
        df[f"lag_{i}"] = df["ret"].shift(i)
    df = df.dropna()
    return df
