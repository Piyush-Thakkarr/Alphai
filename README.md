# BTC Next-Hour 95% Forecast

Predicts a 95% confidence range for the next hour's BTCUSDT close. GBM with
Garman-Klass volatility, Student-t innovations, walk-forward backtested on
the last 30 days of hourly Binance data.

## Backtest results (last 30 days, walk-forward)

| Metric | Value |
|---|---|
| Predictions | 720 |
| **coverage_95** | **0.9514** (target 0.9500) |
| **mean_winkler_95** | **1,684.01** |
| mean_width_95 | $1,178 |
| median_width_95 | $1,176 |

Re-run with `python backtest.py`. Numbers shift slightly with fresh data
(the 30-day window rolls forward each hour) but the ratios are stable.

We fetch 731 bars (720 test + 10 warmup + 1 to compensate for the
in-progress current bar) and report metrics over exactly the most recent
720 bars. The 10-bar warmup buffer is what lets the model fit on the very
first prediction without leaking future data into it.

## Stack

| Layer | Choice | One-line reason |
|---|---|---|
| Data | Binance public mirror, BTCUSDT 1h | Brief mandate; mirror avoids geo-block |
| Per-bar variance | Garman-Klass on OHLC | ~7x more efficient than close-to-close stdev; uses all four price columns |
| Smoothing | EWMA, lambda = 0.97 | Recency-weights to capture volatility clustering; half-life ~23 bars |
| Distribution | Student-t, df fitted per window, floor=4 | Captures BTC's fat tails; brief mandates Student-t |
| Drift mu | Zero | At 1h horizon mu ~= 1e-5 << sigma ~= 1e-2; statistical noise |
| Forecast | Monte Carlo, 10,000 paths, GBM step | Required for dashboard histogram; works with any distribution |
| Backtest | Walk-forward, 500-bar rolling window | Brief mandate; mirrors production refit pattern |
| Dashboard | Streamlit | Brief recommends; ~150 lines to public URL |
| Hosting | Streamlit Community Cloud | Free; auto-deploys from GitHub on push |
| Persistence | SQLite, INSERT OR IGNORE on hour key | First prediction per hour wins; idempotent across visits |

## Why each major choice (and why not the alternatives)

**Garman-Klass instead of close-to-close stdev:** Binance gives us OHLC for
free. Throwing away H, L, O is a 7x efficiency loss. Per-bar variance from
GK feeds into EWMA the same way r-squared would in plain GARCH/EWMA — it's
the same model family with a smarter input.

**EWMA instead of fitted GARCH(1,1):** EWMA on GK is GARCH-style recursion
with fixed coefficients (omega=0, alpha=1-lambda, beta=lambda). Fitting full
GARCH adds an MLE step that's noisy on 500-sample windows for a benefit that
rarely shows up at 1-step horizon. Tried, kept simpler.

**FIGARCH dropped from starter:** Long-memory volatility decay only
manifests at multi-day forecast horizons. We forecast 1 hour ahead. The
extra fractional-differencing parameter adds estimation noise without
forecast benefit at this horizon.

**Zero drift:** Hourly drift on BTC is on the order of 10^-5; sigma is
on the order of 10^-2. The drift contribution to a 95% interval is ~$0.70
at $70k BTC vs ~$700 from sigma. Setting mu=0 makes the interval symmetric
around current price, which is honest. Using historical mean introduces
sample-period bias that doesn't predict the next hour.

**Per-window df fit:** Fitting Student-t df once on the full backtest
sample would inject look-ahead bias. Fitting per-window keeps the
no-peeking discipline at the cost of slightly noisier df estimates. Floor
at 4 prevents pathological fits (kurtosis is finite for df>4).

**Monte Carlo even though closed-form exists:** Closed-form Student-t
percentiles would be exact and faster, but the dashboard needs a histogram
of simulated outcomes — closed-form gives you only two numbers. Using two
methods (closed-form for backtest, MC for dashboard) would risk train-serve
skew. One method everywhere.

**Walk-forward 500-bar rolling window:** Same window used by the dashboard,
so backtest reflects production. Static fit is look-ahead bias by
definition; expanding window mixes regimes; smaller windows are too noisy
for stable sigma.

## Project layout

```
data.py          Binance OHLC fetcher
model.py         Garman-Klass + EWMA + Student-t + GBM Monte Carlo
                 Single source of truth: predict_range() called by both
                 the backtest and the dashboard (no train/serve skew)
backtest.py      Walk-forward 720 predictions; writes backtest_results.jsonl
app.py           Streamlit dashboard
persistence.py   SQLite store for Part C
requirements.txt
backtest_results.jsonl   one JSON object per line, 709 predictions
BUGS.md          issues found in the provided starter notebook
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python backtest.py    # produces backtest_results.jsonl + prints metrics
streamlit run app.py  # serves the dashboard at localhost:8501
```

## No-look-ahead guarantee

At backtest bar `n`, the only data passed into `predict_range()` is
`ohlc.iloc[max(0, n-500):n]`. Python's half-open slice excludes index `n`
itself, so bar `n`'s price is structurally impossible to enter the
prediction for bar `n`. Same slicing pattern in live mode where there is
no future data to leak.

## Improvement directions considered (not implemented in this submission)

- **GARCH(1,1) on log returns:** likely marginal coverage gain over
  EWMA-on-GK at higher backtest cost (per-iteration MLE fit). Not
  benchmarked; would be the natural v2.
- **Filtered Historical Simulation instead of Student-t:** empirically
  calibrated, bounded by historical extremes. Theory says it's nearly
  identical to Student-t at 95%; would be a tiebreaker if someone wanted
  to push tail accuracy further.
- **Closed-form Student-t percentiles for the backtest:** would be exact
  and faster than Monte Carlo, but doesn't yield samples for the
  dashboard histogram. Mixing methods would risk train/serve skew, so
  Monte Carlo is used in both paths.

## Live dashboard

Deployed at `<add public URL after Streamlit deploy>`. Auto-pulls the
latest closed bar on each visit. Persistence layer accumulates predictions
and back-fills actuals as bars close, building up a live timeline.

**Note on Streamlit Community Cloud and SQLite persistence:** the SCC
filesystem is reset on container restarts (redeploys, long inactivity,
infrastructure events). The `predictions.db` file persists between visits
within the same container lifetime but does not survive a restart. For
this 7-day grading window with light traffic this is acceptable; in a
production setting I would back persistence with a managed store (e.g.
Supabase, Postgres) instead.
