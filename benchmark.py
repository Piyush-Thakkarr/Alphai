from __future__ import annotations

"""
Full grid benchmark: 6 volatility estimators x 3 innovation distributions
on the same 720-bar walk-forward.

vol estimators:
  - rolling stdev (close-to-close, no recency weighting)
  - EWMA on r^2     (close-to-close, RiskMetrics-style)
  - Parkinson + EWMA   (uses H, L per bar)
  - Rogers-Satchell + EWMA (uses O, H, L, C per bar; drift-aware)
  - Garman-Klass + EWMA (uses O, H, L, C per bar; our primary)
  - GARCH(1,1) on log returns

distributions:
  - student-t (df fitted per window, floor=4)
  - FHS (bootstrap from window's standardized residuals)
  - normal (sanity check: brief says it should under-cover)

everything else fixed: 500-bar rolling window, mu=0, 10k MC, no peeking.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from arch import arch_model

from data import fetch_klines
from model import garman_klass_per_bar, EWMA_LAMBDA
from backtest import winkler_score, WINDOW, TEST_BARS, WARMUP_BARS, ALPHA

ROLLING_VAR_WINDOW = 60   # for plain rolling stdev


# ---------- volatility estimators ----------
# Each returns (sigma_next, sigma_series).
# sigma_next: scalar, the sigma used to predict the next bar (after the input).
# sigma_series: pd.Series indexed by ohlc.index, sigma_t for each bar
#   (the sigma that WOULD have predicted bar t, based on data <= t-1).
#   NaN where unavailable. Used by FHS for standardizing residuals.

def vol_rolling_stdev(ohlc):
    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1))
    rolled = log_ret.rolling(ROLLING_VAR_WINDOW).std(ddof=1)
    sigma_series = rolled.shift(1)  # sigma_t uses returns up to t-1
    sigma_next = float(rolled.iloc[-1])
    return sigma_next, sigma_series


def vol_ewma_r2(ohlc):
    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1)).fillna(0.0)
    sigma2 = (log_ret ** 2).ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    sigma_next = float(np.sqrt(sigma2.iloc[-1]))
    sigma_series = np.sqrt(sigma2.shift(1))
    return sigma_next, sigma_series


def vol_parkinson_ewma(ohlc):
    park = (np.log(ohlc["high"] / ohlc["low"])) ** 2 / (4.0 * np.log(2.0))
    sigma2 = park.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    sigma_next = float(np.sqrt(sigma2.iloc[-1]))
    sigma_series = np.sqrt(sigma2.shift(1))
    return sigma_next, sigma_series


def vol_rogers_satchell_ewma(ohlc):
    h, l, c, o = ohlc["high"], ohlc["low"], ohlc["close"], ohlc["open"]
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    sigma2 = rs.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    sigma_next = float(np.sqrt(sigma2.iloc[-1]))
    sigma_series = np.sqrt(sigma2.shift(1))
    return sigma_next, sigma_series


def vol_garman_klass_ewma(ohlc):
    """Our primary. Same as model.ewma_sigma but also returns the series."""
    gk = garman_klass_per_bar(ohlc)
    sigma2 = gk.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    sigma_next = float(np.sqrt(sigma2.iloc[-1]))
    sigma_series = np.sqrt(sigma2.shift(1))
    return sigma_next, sigma_series


def vol_garch11(ohlc):
    log_ret_pct = (np.log(ohlc["close"] / ohlc["close"].shift(1)).dropna()) * 100.0
    am = arch_model(log_ret_pct, vol="GARCH", p=1, q=1, dist="studentst", mean="zero")
    res = am.fit(disp="off", show_warning=False)
    forecast = res.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.iloc[-1, 0])) / 100.0
    # arch's conditional_volatility is on the log_ret index (drops first NaN row)
    cond_vol = res.conditional_volatility / 100.0
    sigma_series = pd.Series(index=ohlc.index, dtype=float)
    sigma_series.loc[cond_vol.index] = cond_vol.values
    return sigma_next, sigma_series


# ---------- innovation distributions ----------

def shock_student_t(sigma_next, sigma_series, ohlc, rng, n_sims):
    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1)).dropna()
    standardized = (log_ret - log_ret.mean()) / log_ret.std(ddof=1)
    df_fit, _, _ = stats.t.fit(standardized, floc=0.0, fscale=1.0)
    df_t = max(float(df_fit), 4.0)
    raw_t = rng.standard_t(df_t, size=n_sims)
    z = raw_t * np.sqrt((df_t - 2.0) / df_t)
    return z, df_t


def shock_fhs(sigma_next, sigma_series, ohlc, rng, n_sims):
    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1))
    standardized = (log_ret / sigma_series).dropna().values
    if len(standardized) < 30:
        raise ValueError(f"FHS needs >=30 residuals, got {len(standardized)}")
    z = rng.choice(standardized, size=n_sims, replace=True)
    return z, float("nan")


def shock_normal(sigma_next, sigma_series, ohlc, rng, n_sims):
    z = rng.standard_normal(size=n_sims)
    return z, float("inf")  # df=inf is the normal distribution


# ---------- combined predictor ----------

def predict(ohlc, vol_fn, shock_fn, alpha=ALPHA, n_sims=10_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sigma_next, sigma_series = vol_fn(ohlc)
    z, _df = shock_fn(sigma_next, sigma_series, ohlc, rng, n_sims)
    log_returns = -0.5 * sigma_next ** 2 + sigma_next * z
    current_price = float(ohlc["close"].iloc[-1])
    samples = current_price * np.exp(log_returns)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    low, high = np.percentile(samples, [lo_q, hi_q])
    return {"low": float(low), "high": float(high), "sigma": sigma_next}


# ---------- walk-forward harness ----------

def run(ohlc, vol_fn, shock_fn, label, window=WINDOW, alpha=ALPHA):
    rng = np.random.default_rng(42)
    coverage_hits, widths, winklers = 0, [], []
    n_bars = len(ohlc)
    test_start = max(1, n_bars - TEST_BARS)
    skipped = 0
    for n in tqdm(range(test_start, n_bars), desc=label, leave=False):
        info_set = ohlc.iloc[max(0, n - window):n]
        try:
            p = predict(info_set, vol_fn, shock_fn, alpha=alpha, rng=rng)
        except Exception:
            skipped += 1
            continue
        actual = float(ohlc["close"].iloc[n])
        in_range = p["low"] <= actual <= p["high"]
        coverage_hits += int(in_range)
        widths.append(p["high"] - p["low"])
        winklers.append(winkler_score(p["low"], p["high"], actual, alpha))
    n_pred = len(widths)
    return {
        "label": label,
        "n": n_pred,
        "skipped": skipped,
        "coverage": coverage_hits / n_pred if n_pred else float("nan"),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "mean_winkler": float(np.mean(winklers)) if winklers else float("nan"),
        "median_width": float(np.median(widths)) if widths else float("nan"),
        "median_winkler": float(np.median(winklers)) if winklers else float("nan"),
    }


# ---------- the grid ----------

VOL_METHODS = [
    ("rolling-stdev",   vol_rolling_stdev),
    ("ewma-r2",         vol_ewma_r2),
    ("parkinson-ewma",  vol_parkinson_ewma),
    ("rogers-sat-ewma", vol_rogers_satchell_ewma),
    ("GK-ewma",         vol_garman_klass_ewma),    # our primary
    ("garch(1,1)",      vol_garch11),
]

SHOCK_METHODS = [
    ("student-t", shock_student_t),
    ("fhs",       shock_fhs),
    ("normal",    shock_normal),
]


if __name__ == "__main__":
    fetch_count = TEST_BARS + WARMUP_BARS + 1
    print(f"fetching {fetch_count} bars from binance...")
    ohlc = fetch_klines(limit=fetch_count)
    print(f"got {len(ohlc)} closed bars\n")

    print(f"running {len(VOL_METHODS)} x {len(SHOCK_METHODS)} = "
          f"{len(VOL_METHODS) * len(SHOCK_METHODS)} arms over {TEST_BARS} bars each\n")

    results = []
    for v_label, v_fn in VOL_METHODS:
        for s_label, s_fn in SHOCK_METHODS:
            label = f"{v_label} + {s_label}"
            r = run(ohlc, v_fn, s_fn, label)
            results.append(r)
            print(f"  {label:<30} cov={r['coverage']:.4f}  "
                  f"width=${r['mean_width']:>8,.0f}  winkler={r['mean_winkler']:>9,.1f}  "
                  f"n={r['n']}{' (skipped %d)' % r['skipped'] if r['skipped'] else ''}")

    # ---------- ranking ----------
    print("\n" + "=" * 90)
    print("FULL TABLE (sorted by mean Winkler, lower is better)")
    print("=" * 90)
    print(f"{'rank':<5} {'arm':<32} {'cov':>8} {'meanW':>10} {'mednW':>10} "
          f"{'meanWidth':>11} {'n':>5}")
    print("-" * 90)
    for i, r in enumerate(sorted(results, key=lambda x: x["mean_winkler"]), 1):
        print(f"{i:<5} {r['label']:<32} {r['coverage']:>8.4f} {r['mean_winkler']:>10,.1f} "
              f"{r['median_winkler']:>10,.1f} ${r['mean_width']:>10,.0f} {r['n']:>5}")

    print("\n" + "=" * 90)
    print("FILTERED to coverage in [0.93, 0.97] (calibrated arms only)")
    print("=" * 90)
    calibrated = [r for r in results if 0.93 <= r["coverage"] <= 0.97]
    if calibrated:
        for i, r in enumerate(sorted(calibrated, key=lambda x: x["mean_winkler"]), 1):
            print(f"{i:<5} {r['label']:<32} {r['coverage']:>8.4f} "
                  f"{r['mean_winkler']:>10,.1f} ${r['mean_width']:>10,.0f}")
    else:
        print("  (none)")
