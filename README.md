# btc next-hour 95% forecast

predicts the 95% range for the next hour's BTCUSDT close. GBM with
Garman-Klass volatility from OHLC, EWMA smoothing, Student-t innovations.
walk-forward backtested on the last 30 days of binance hourly bars.

live dashboard: `<paste public URL after streamlit deploy>`

## numbers from the backtest

- 720 predictions
- coverage_95: **0.9556** (target 0.95)
- mean_winkler_95: **1685.78**
- mean width: $1,184

re-run with `python backtest.py`. the 30-day window slides forward each
hour so the numbers drift slightly run-to-run.

## stack

- data: binance public mirror (`data-api.binance.vision`), BTCUSDT 1h
- per-bar variance: garman-klass from O/H/L/C (~7x more efficient than
  close-to-close stdev)
- smoothing: EWMA, lambda = 0.97
- distribution: Student-t, df fitted per window, floored at 4
- drift mu: 0
- forecast: monte carlo, 10,000 paths, GBM step
- backtest: walk-forward, 500-bar rolling window
- dashboard: streamlit on streamlit community cloud
- persistence (part C): sqlite

## files

- `data.py` — binance fetcher
- `model.py` — `predict_range()`. one function, called by both
  the backtest and the dashboard so they always agree
- `backtest.py` — walks forward 720 bars, writes
  `backtest_results.jsonl`
- `app.py` — the streamlit dashboard
- `persistence.py` — sqlite for part C
- `BUGS.md` — stuff i caught while reading the starter

## running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backtest.py
streamlit run app.py
```
