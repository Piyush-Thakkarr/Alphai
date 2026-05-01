# BTC Next-Hour 95% Forecast

Predicts a 95% range for the next hour's BTCUSDT close. GBM with
Garman-Klass volatility, Student-t innovations. Walk-forward backtested
on the last 30 days of hourly Binance data.

**Live dashboard:** `<paste public URL after Streamlit deploy>`

## Backtest results

| Metric | Value |
|---|---|
| Predictions | 720 |
| **coverage_95** | **0.9514** (target 0.9500) |
| **mean_winkler_95** | **1,684.01** |
| mean_width_95 | $1,178 |

## Stack

| Layer | Choice |
|---|---|
| Data | Binance public mirror, BTCUSDT 1h |
| Per-bar variance | Garman-Klass (OHLC) |
| Smoothing | EWMA, lambda = 0.97 |
| Distribution | Student-t, df fitted per window, floor=4 |
| Drift mu | 0 |
| Forecast | Monte Carlo, 10,000 paths |
| Backtest | Walk-forward, 500-bar rolling window |
| Dashboard | Streamlit + Streamlit Community Cloud |
| Persistence (Part C) | SQLite |

## Files

```
data.py          Binance OHLC fetcher
model.py         predict_range(): GK + EWMA + Student-t + GBM Monte Carlo
backtest.py      Walk-forward 720 predictions -> backtest_results.jsonl
app.py           Streamlit dashboard
persistence.py   SQLite store
BUGS.md          Issues found in the provided starter notebook
```

`predict_range()` in `model.py` is called by both the backtest and the
dashboard — same numbers in both places by construction.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backtest.py     # writes backtest_results.jsonl + prints metrics
streamlit run app.py   # serves the dashboard at localhost:8501
```
