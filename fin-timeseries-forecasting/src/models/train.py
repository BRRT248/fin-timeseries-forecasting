
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import load_prices, to_returns, make_features
from src.utils.walkforward import expanding_windows
from src.models.arima_model import ARIMAWrap
from src.models.lstm_model import LSTMWrap

def main(ticker, start, end, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    prices = load_prices(ticker, start, end)
    rets = to_returns(prices)
    df = make_features(rets, lags=5)

    y = df["ret"].values
    n = len(df)
    initial = int(n * 0.6)
    if initial < 30:
        initial = max(20, n//2)

    preds_arima, preds_lstm, actuals = [], [], []

    for tr, te in expanding_windows(n, initial, step=1):
        y_tr = y[tr]
        ar = ARIMAWrap(order=(1,0,1)).fit(y_tr)
        pa = ar.predict_next()

        lstm = LSTMWrap(window=20, epochs=6, batch_size=32, lr=1e-3).fit(y_tr)
        pl = lstm.predict_next()

        preds_arima.append(pa)
        preds_lstm.append(pl)
        actuals.append(y[te][0])

    res = pd.DataFrame({
        "actual": actuals,
        "pred_arima": preds_arima,
        "pred_lstm": preds_lstm
    }, index=df.index[-len(actuals):])
    res.to_csv(outdir / "oos_predictions.csv", index=True)

    res["pred_combo"] = res[["pred_arima", "pred_lstm"]].mean(axis=1)
    res["signal"] = np.sign(res["pred_combo"]).replace(0, 0)

    tc = 0.0002
    res["signal_shift"] = res["signal"].shift(1).fillna(0)
    res["turnover"] = (res["signal"] != res["signal_shift"]).astype(int)
    res["strategy_ret"] = res["signal"] * res["actual"] - tc * res["turnover"]

    res.to_csv(outdir / "backtest.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--outdir", type=str, default="artifacts")
    args = p.parse_args()
    main(args.ticker, args.start, args.end, args.outdir)
