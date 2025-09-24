
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.metrics import sharpe_ratio, max_drawdown, hit_rate

def plot_forecast_vs_actual(df, outpath):
    plt.figure()
    df["actual"].rolling(5).mean().plot(label="actual (5d MA)")
    df["pred_arima"].rolling(5).mean().plot(label="pred_arima (5d MA)")
    df["pred_lstm"].rolling(5).mean().plot(label="pred_lstm (5d MA)")
    plt.title("Out-of-sample: Forecast vs Actual (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_equity(df, outpath):
    equity = (1 + df["strategy_ret"]).cumprod()
    peak = np.maximum.accumulate(equity.values)
    drawdown = equity.values/peak - 1
    plt.figure()
    plt.plot(equity.index, equity.values, label="Equity")
    plt.plot(equity.index, drawdown, label="Drawdown")
    plt.title("Equity Curve & Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_rolling_metrics(df, outpath):
    r = df["strategy_ret"]
    window = 60
    rolling_sharpe = r.rolling(window).apply(sharpe_ratio, raw=False)
    rolling_hit = r.rolling(window).apply(hit_rate, raw=False)
    plt.figure()
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, label="Rolling Sharpe (60d)")
    plt.plot(rolling_hit.index, rolling_hit.values, label="Rolling Hit-rate (60d)")
    plt.title("Rolling Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(artifacts, outdir):
    artifacts = Path(artifacts)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(artifacts / "backtest.csv", parse_dates=True, index_col=0)

    sr = sharpe_ratio(df["strategy_ret"])
    mdd = max_drawdown((1+df["strategy_ret"]).cumprod())
    hr = df["strategy_ret"].gt(0).mean()

    with open(outdir / "summary.txt", "w") as f:
        f.write(f"Sharpe: {sr:.3f}\nMax Drawdown: {mdd:.3%}\nHit-rate: {hr:.2%}\n")

    plot_forecast_vs_actual(df, outdir / "forecast_vs_actual.png")
    plot_equity(df, outdir / "equity_curve.png")
    plot_rolling_metrics(df, outdir / "rolling_metrics.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--outdir", type=str, default="figures")
    args = p.parse_args()
    main(args.artifacts, args.outdir)
