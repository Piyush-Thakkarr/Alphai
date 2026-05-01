# btc next-hour 95% forecast

predicts the 95% range for the next hour's BTCUSDT close. GBM with
Garman-Klass volatility from OHLC, EWMA smoothing, Student-t innovations.
walk-forward backtested on the last 30 days of binance hourly bars.

live dashboard: `<paste public URL after streamlit deploy>`

## numbers from the backtest

- 720 predictions
- coverage_95: **0.9556** (target 0.95)
- mean_winkler_95: **1702.85**
- mean width: $1,197

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

## benchmarks vs alternatives

before settling on EWMA-GK + student-t, i ran a 3-arm benchmark on the
same 720-bar walk-forward. same MC, same 500-bar window, mu=0. only
the volatility model and/or innovation distribution differ.

run `python benchmark.py` to reproduce.

| metric          | EWMA-GK + t (ours) | EWMA-GK + FHS | GARCH(1,1) + t |
|---|---|---|---|
| predictions     | **720**           | 699           | 720             |
| coverage @ 95%  | **0.9556**        | 0.9599        | 0.9403          |
| mean width      | $1,184            | $1,198        | $1,194          |
| mean winkler    | 1,686             | **1,683**     | 1,718           |
| median width    | $1,188            | $1,182        | $1,115          |
| median winkler  | 1,204             | 1,210         | 1,145           |

**GARCH(1,1) + student-t**: under-covers (0.94 vs target 0.95). more
misses, and the winkler penalty on misses (2/alpha = 40 times distance
outside the band) pushes mean winkler higher despite tighter typical
bars. eliminated.

**EWMA-GK + FHS**: marginally lower mean winkler (1,683 vs 1,686, a
~0.2% difference). but FHS only covered 699 of the 720 bars because
the first ~20 test bars don't have enough standardized residuals to
bootstrap from (FHS needs >=30 to be stable). not a like-for-like
comparison.

**verdict**: keep EWMA-GK + student-t. it predicts all 720 bars,
calibrates closest to 0.95, and the FHS Winkler edge falls within
run-to-run noise (the 30-day window slides every hour, numbers drift
by ~30 winkler between reruns).
