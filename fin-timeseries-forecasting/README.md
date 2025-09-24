
# Financial Time Series Forecasting (ARIMA vs LSTM) — Walk-Forward Backtest

Advanced, production-style repo demonstrating rigorous **time series modelling** and **evaluation** with:
- **ARIMA baseline** (statsmodels) vs **LSTM** (TensorFlow/Keras) on returns
- **Walk-forward** expanding window backtest
- **Forecast combination** (simple average) option
- **Transaction costs**, **equity curve**, **max drawdown**, and **rolling metrics**
- **Deterministic seeding**, modular code, tests, and plots saved to `figures/`

## Problem
Forecast next-period returns and assess strategy profitability using a **trading rule** driven by model forecasts, with realistic walk-forward validation and transaction costs.

## Data
- If `yfinance` is available, pulls daily close prices for a specified ticker (default `SPY`).
- Otherwise generates **synthetic geometric Brownian motion** prices.
- Data prepped into **log returns** (stationary) with lagged features.

## Methods
- **ARIMA(p, d=0, q)** on returns (stationary by construction).
- **LSTM** on windowed return sequences.
- **Walk-forward** expanding window with re-fit each step to avoid look-ahead.
- Strategy: go long if forecast > 0, short if forecast < 0 (simple sign strategy).

## Results & Visualisations
- `figures/forecast_vs_actual.png` — Out-of-sample forecasts vs actual returns (smoothed)
- `figures/equity_curve.png` — Equity curve with drawdown overlay
- `figures/rolling_metrics.png` — Rolling Sharpe & hit-rate

> All plots use **matplotlib** only and keep styles simple for reproducibility.

## Reproducibility
```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
make train
make eval
```

## Limitations
- LSTM hyperparams are modest to keep runtimes reasonable.
- Transaction costs assumed constant (configurable).

## License
MIT
