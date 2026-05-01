from __future__ import annotations

"""
Empirical sweet-spot search: at which intra-bar granularity does
realized variance give the best Winkler at calibrated coverage?

Strategy:
  1. Pull 31 days of 1-minute bars (paginated)
  2. Sub-sample to K-min granularity for K in {1, 2, 3, 5, 10, 15, 30}
  3. Compute hourly realized variance from K-min squared log returns
  4. Run the same walk-forward backtest for each granularity
  5. Compare to our current Garman-Klass-on-1h baseline

Same student-t innovations, same 500-bar window, same MC, same mu=0.
Only the per-bar variance estimator changes.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from data import fetch_klines
from data_intra import fetch_klines_paginated
from model import garman_klass_per_bar, EWMA_LAMBDA
from backtest import winkler_score, WINDOW, TEST_BARS, WARMUP_BARS, ALPHA

WARMUP_DAYS_INTRA = 1   # how much extra 1m data to fetch before the 30-day test set
TEST_DAYS = 30


# ---------- realized variance ----------

def hourly_rv_from_1min(close_1m: pd.Series, K_minutes: int) -> pd.Series:
    """
    Sub-sample 1m closes to K-min, compute log returns, sum squared
    returns within each hour. Returns a Series indexed by hour close_time
    (end of hour, like binance's 1h bar close_time).
    """
    # sub-sample: take every K-th close
    close_K = close_1m.iloc[::K_minutes]
    log_ret_K = np.log(close_K / close_K.shift(1)).dropna()

    # group by hour (using the start of the hour as group key)
    hour_start = log_ret_K.index.floor("1h")
    rv_by_hour_start = (log_ret_K ** 2).groupby(hour_start).sum()

    # shift index from hour-start to hour-close_time (matches binance 1h bars)
    rv_by_hour_start.index = (
        rv_by_hour_start.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    )
    return rv_by_hour_start


# ---------- volatility from a precomputed RV series ----------

def make_vol_fn_from_rv(rv_series: pd.Series, label: str):
    """Returns a vol_fn(ohlc) -> (sigma_next, sigma_series) using a
    precomputed hourly RV series."""
    def vol_fn(ohlc):
        # Align RV to the bars in this ohlc slice
        rv_bars = rv_series.reindex(ohlc.index)
        # EWMA-smooth across hours
        sigma2 = rv_bars.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
        sigma_next = float(np.sqrt(sigma2.iloc[-1]))
        sigma_series = np.sqrt(sigma2.shift(1))
        return sigma_next, sigma_series
    vol_fn.__name__ = f"rv_{label}"
    return vol_fn


def vol_garman_klass_ewma(ohlc):
    """Baseline: per-bar Garman-Klass + EWMA (what we ship)."""
    gk = garman_klass_per_bar(ohlc)
    sigma2 = gk.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    sigma_next = float(np.sqrt(sigma2.iloc[-1]))
    sigma_series = np.sqrt(sigma2.shift(1))
    return sigma_next, sigma_series


# ---------- predictor (student-t innovations, same as ours) ----------

def predict(ohlc, vol_fn, alpha=ALPHA, n_sims=10_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sigma_next, _sigma_series = vol_fn(ohlc)
    if not np.isfinite(sigma_next) or sigma_next <= 0:
        raise ValueError(f"bad sigma: {sigma_next}")

    # student-t df fitted on this window
    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1)).dropna()
    standardized = (log_ret - log_ret.mean()) / log_ret.std(ddof=1)
    df_fit, _, _ = stats.t.fit(standardized, floc=0.0, fscale=1.0)
    df_t = max(float(df_fit), 4.0)
    raw_t = rng.standard_t(df_t, size=n_sims)
    z = raw_t * np.sqrt((df_t - 2.0) / df_t)

    log_returns = -0.5 * sigma_next ** 2 + sigma_next * z
    current_price = float(ohlc["close"].iloc[-1])
    samples = current_price * np.exp(log_returns)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"low": float(lo), "high": float(hi)}


def run_walkforward(ohlc, vol_fn, label):
    rng = np.random.default_rng(42)
    coverage_hits, widths, winklers = 0, [], []
    skipped = 0
    n_bars = len(ohlc)
    test_start = max(1, n_bars - TEST_BARS)
    for n in tqdm(range(test_start, n_bars), desc=label, leave=False):
        info_set = ohlc.iloc[max(0, n - WINDOW):n]
        try:
            p = predict(info_set, vol_fn, alpha=ALPHA, rng=rng)
        except Exception:
            skipped += 1
            continue
        actual = float(ohlc["close"].iloc[n])
        in_range = p["low"] <= actual <= p["high"]
        coverage_hits += int(in_range)
        widths.append(p["high"] - p["low"])
        winklers.append(winkler_score(p["low"], p["high"], actual, ALPHA))
    n_pred = len(widths)
    return {
        "label": label,
        "n": n_pred,
        "skipped": skipped,
        "coverage": coverage_hits / n_pred if n_pred else float("nan"),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "median_width": float(np.median(widths)) if widths else float("nan"),
        "mean_winkler": float(np.mean(winklers)) if winklers else float("nan"),
        "median_winkler": float(np.median(winklers)) if winklers else float("nan"),
    }


GRANULARITIES = [1, 2, 3, 5, 10, 15, 30]


if __name__ == "__main__":
    # 1) Fetch 1h bars for the test set (same as our backtest)
    fetch_count = TEST_BARS + WARMUP_BARS + 1
    print(f"[1/3] fetching {fetch_count} 1h bars for the test set...")
    ohlc_1h = fetch_klines(limit=fetch_count)
    print(f"      got {len(ohlc_1h)} closed 1h bars")

    # 2) Fetch enough 1-min bars to cover all test 1h bars + extra
    one_min_bars_needed = (TEST_DAYS + WARMUP_DAYS_INTRA) * 24 * 60
    print(f"[2/3] fetching {one_min_bars_needed} 1-minute bars (paginated)...")
    df_1m = fetch_klines_paginated("BTCUSDT", "1m", one_min_bars_needed)
    print(f"      got {len(df_1m)} closed 1m bars covering "
          f"{df_1m.index[0]}  ->  {df_1m.index[-1]}")
    close_1m = df_1m["close"]

    # 3) Run baseline + each granularity
    print(f"\n[3/3] running walk-forward over {TEST_BARS} bars per arm")
    print(f"      arms: 1 baseline (Garman-Klass) + {len(GRANULARITIES)} RV granularities\n")

    results = []
    r0 = run_walkforward(ohlc_1h, vol_garman_klass_ewma, "GK-ewma (ours, baseline)")
    results.append(r0)
    print(f"  {r0['label']:<32} cov={r0['coverage']:.4f}  "
          f"width=${r0['mean_width']:>8,.0f}  winkler={r0['mean_winkler']:>9,.1f}  n={r0['n']}")

    for K in GRANULARITIES:
        rv = hourly_rv_from_1min(close_1m, K_minutes=K)
        # only use the portion of RV that aligns with our 1h test bars
        vol_fn = make_vol_fn_from_rv(rv, label=f"{K}m")
        label = f"realized-var ({K}m)"
        r = run_walkforward(ohlc_1h, vol_fn, label)
        results.append(r)
        print(f"  {r['label']:<32} cov={r['coverage']:.4f}  "
              f"width=${r['mean_width']:>8,.0f}  winkler={r['mean_winkler']:>9,.1f}  n={r['n']}")

    # ---------- summary ----------
    print("\n" + "=" * 80)
    print("SORTED BY MEAN WINKLER (lower = better)")
    print("=" * 80)
    print(f"{'rank':<5} {'arm':<32} {'cov':>8} {'meanW':>10} {'mednW':>10} {'width':>10} {'n':>5}")
    print("-" * 80)
    for i, r in enumerate(sorted(results, key=lambda x: x["mean_winkler"]), 1):
        print(f"{i:<5} {r['label']:<32} {r['coverage']:>8.4f} {r['mean_winkler']:>10,.1f} "
              f"{r['median_winkler']:>10,.1f} ${r['mean_width']:>8,.0f} {r['n']:>5}")

    print("\n" + "=" * 80)
    print("CALIBRATED ARMS (coverage in [0.93, 0.97]) sorted by mean Winkler")
    print("=" * 80)
    cal = [r for r in results if 0.93 <= r["coverage"] <= 0.97]
    if cal:
        for i, r in enumerate(sorted(cal, key=lambda x: x["mean_winkler"]), 1):
            print(f"{i:<5} {r['label']:<32} {r['coverage']:>8.4f} "
                  f"{r['mean_winkler']:>10,.1f} ${r['mean_width']:>8,.0f}")

    # the actual sweet spot
    print()
    cal_sorted = sorted(cal, key=lambda x: x["mean_winkler"]) if cal else []
    if cal_sorted:
        winner = cal_sorted[0]
        print(f"-> empirical sweet spot: {winner['label']}  "
              f"(cov {winner['coverage']:.4f}, winkler {winner['mean_winkler']:,.1f})")
        baseline = next((r for r in results if "GK-ewma" in r["label"]), None)
        if baseline and winner["label"] != baseline["label"]:
            delta = baseline["mean_winkler"] - winner["mean_winkler"]
            pct = delta / baseline["mean_winkler"] * 100
            print(f"   vs baseline GK-ewma: {delta:+.1f} winkler ({pct:+.2f}%)")
