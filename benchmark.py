from __future__ import annotations

"""
Head-to-head benchmark of three predictors on the same 720-bar
walk-forward backtest:

    A) EWMA-GK + student-t  (what we ship)
    B) EWMA-GK + FHS        (filtered historical simulation)
    C) GARCH(1,1) + student-t

A vs B isolates the innovation distribution (same volatility).
A vs C isolates the volatility model (same innovations).

Same MC, same 500-bar window, mu=0, no peeking.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from arch import arch_model

from data import fetch_klines
from model import predict_range as predict_ewma_gk
from model import garman_klass_per_bar, EWMA_LAMBDA
from backtest import winkler_score, WINDOW, TEST_BARS, WARMUP_BARS, ALPHA


def predict_range_fhs(
    ohlc: pd.DataFrame,
    alpha: float = ALPHA,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Same EWMA-GK volatility, but instead of student-t shocks, sample
    standardized residuals from this window with replacement.

    standardized residual at bar k = r_k / sigma_k
    where sigma_k = EWMA-GK using data up to bar k-1.
    """
    if rng is None:
        rng = np.random.default_rng()

    gk = garman_klass_per_bar(ohlc)
    # ewm value at bar k uses GK up to and including bar k
    ewm_series = gk.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    # for predicting bar k, we use sigma based on GK up to k-1
    sigma_at_each_bar = np.sqrt(ewm_series.shift(1))

    log_ret = np.log(ohlc["close"] / ohlc["close"].shift(1))
    standardized = (log_ret / sigma_at_each_bar).dropna().values

    if len(standardized) < 30:
        raise ValueError(f"FHS needs >=30 standardized residuals, got {len(standardized)}")

    # sigma for predicting the NEXT bar (uses GK up to and incl. last input bar)
    sigma_next = float(np.sqrt(ewm_series.iloc[-1]))

    # bootstrap with replacement from the residual bag
    z = rng.choice(standardized, size=n_sims, replace=True)
    log_returns = -0.5 * sigma_next ** 2 + sigma_next * z

    current_price = float(ohlc["close"].iloc[-1])
    samples = current_price * np.exp(log_returns)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    low, high = np.percentile(samples, [lo_q, hi_q])

    return {
        "low": float(low),
        "high": float(high),
        "sigma": sigma_next,
        "df_t": float("nan"),  # FHS has no df
        "samples": samples,
        "current_price": current_price,
    }


def predict_range_garch(
    ohlc: pd.DataFrame,
    alpha: float = ALPHA,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Same shape as predict_range, but sigma comes from GARCH(1,1) and
    df comes from arch's t-distribution fit. Innovations are still
    rescaled student-t. Same MC, same percentile read.
    """
    if rng is None:
        rng = np.random.default_rng()

    # arch wants returns scaled by 100, mean='zero' to match our mu=0
    log_ret_pct = (np.log(ohlc["close"] / ohlc["close"].shift(1)).dropna()) * 100.0

    am = arch_model(log_ret_pct, vol="GARCH", p=1, q=1, dist="studentst", mean="zero")
    res = am.fit(disp="off", show_warning=False)

    forecast = res.forecast(horizon=1, reindex=False)
    sigma2_next_pct2 = float(forecast.variance.iloc[-1, 0])
    sigma = np.sqrt(sigma2_next_pct2) / 100.0   # un-scale back to fractional

    df_t = max(float(res.params["nu"]), 4.0)

    raw_t = rng.standard_t(df_t, size=n_sims)
    z = raw_t * np.sqrt((df_t - 2.0) / df_t)
    log_returns = -0.5 * sigma ** 2 + sigma * z

    current_price = float(ohlc["close"].iloc[-1])
    samples = current_price * np.exp(log_returns)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    low, high = np.percentile(samples, [lo_q, hi_q])

    return {
        "low": float(low),
        "high": float(high),
        "sigma": sigma,
        "df_t": df_t,
        "samples": samples,
        "current_price": current_price,
    }


def run(ohlc: pd.DataFrame, predict_fn, label: str) -> dict:
    """Run the walk-forward backtest with the given predictor."""
    rng = np.random.default_rng(42)
    coverage_hits, widths, winklers = 0, [], []

    n_bars = len(ohlc)
    test_start = max(1, n_bars - TEST_BARS)

    for n in tqdm(range(test_start, n_bars), desc=label):
        info_set = ohlc.iloc[max(0, n - WINDOW):n]
        try:
            p = predict_fn(info_set, alpha=ALPHA, rng=rng)
        except Exception as e:
            # GARCH MLE can occasionally fail on a window; skip those
            print(f"  [{label}] skipped bar {n}: {e}")
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
        "coverage": coverage_hits / n_pred if n_pred else float("nan"),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "median_width": float(np.median(widths)) if widths else float("nan"),
        "mean_winkler": float(np.mean(winklers)) if winklers else float("nan"),
        "median_winkler": float(np.median(winklers)) if winklers else float("nan"),
    }


if __name__ == "__main__":
    fetch_count = TEST_BARS + WARMUP_BARS + 1
    print(f"fetching {fetch_count} bars...")
    ohlc = fetch_klines(limit=fetch_count)
    print(f"got {len(ohlc)} closed bars\n")

    a = run(ohlc, predict_ewma_gk, "EWMA-GK + t")
    b = run(ohlc, predict_range_fhs, "EWMA-GK + FHS")
    c = run(ohlc, predict_range_garch, "GARCH + t")

    print("\n" + "=" * 80)
    print(f"{'metric':<20} {'EWMA-GK + t':>18} {'EWMA-GK + FHS':>18} {'GARCH + t':>18}")
    print("-" * 80)
    print(f"{'predictions':<20} {a['n']:>18} {b['n']:>18} {c['n']:>18}")
    print(f"{'coverage @ 95%':<20} {a['coverage']:>18.4f} {b['coverage']:>18.4f} {c['coverage']:>18.4f}")
    print(f"{'mean width ($)':<20} {a['mean_width']:>18,.2f} {b['mean_width']:>18,.2f} {c['mean_width']:>18,.2f}")
    print(f"{'median width ($)':<20} {a['median_width']:>18,.2f} {b['median_width']:>18,.2f} {c['median_width']:>18,.2f}")
    print(f"{'mean winkler':<20} {a['mean_winkler']:>18,.2f} {b['mean_winkler']:>18,.2f} {c['mean_winkler']:>18,.2f}")
    print(f"{'median winkler':<20} {a['median_winkler']:>18,.2f} {b['median_winkler']:>18,.2f} {c['median_winkler']:>18,.2f}")
    print("=" * 80)

    # closest-to-0.95 coverage
    print()
    by_cov = sorted([a, b, c], key=lambda r: abs(r["coverage"] - 0.95))
    print(f"-> closest to 0.95 coverage: {by_cov[0]['label']} "
          f"({by_cov[0]['coverage']:.4f})")

    # lowest mean winkler
    by_wink = sorted([a, b, c], key=lambda r: r["mean_winkler"])
    print(f"-> lowest mean winkler:      {by_wink[0]['label']} "
          f"({by_wink[0]['mean_winkler']:,.2f})")
