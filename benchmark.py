from __future__ import annotations

"""
Head-to-head benchmark of two volatility estimators on the same 720-bar
walk-forward backtest:

    A) EWMA-smoothed Garman-Klass per-bar variance (what we ship)
    B) GARCH(1,1) on log returns (the canonical alternative)

Everything else is identical: student-t innovations, 10k MC, mu=0,
500-bar rolling window, no peeking.

Output: side-by-side coverage / mean width / mean winkler so we can
defend the choice with numbers, not vibes.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from arch import arch_model

from data import fetch_klines
from model import predict_range as predict_ewma_gk
from backtest import winkler_score, WINDOW, TEST_BARS, WARMUP_BARS, ALPHA


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

    a = run(ohlc, predict_ewma_gk, "EWMA-GK")
    b = run(ohlc, predict_range_garch, "GARCH(1,1)")

    print("\n" + "=" * 70)
    print(f"{'metric':<22} {'EWMA-GK':>20} {'GARCH(1,1)':>20}")
    print("-" * 70)
    print(f"{'predictions':<22} {a['n']:>20} {b['n']:>20}")
    print(f"{'coverage @ 95%':<22} {a['coverage']:>20.4f} {b['coverage']:>20.4f}")
    print(f"{'mean width ($)':<22} {a['mean_width']:>20,.2f} {b['mean_width']:>20,.2f}")
    print(f"{'median width ($)':<22} {a['median_width']:>20,.2f} {b['median_width']:>20,.2f}")
    print(f"{'mean winkler':<22} {a['mean_winkler']:>20,.2f} {b['mean_winkler']:>20,.2f}")
    print(f"{'median winkler':<22} {a['median_winkler']:>20,.2f} {b['median_winkler']:>20,.2f}")
    print("=" * 70)

    # Verdict
    print()
    if abs(a["coverage"] - 0.95) < abs(b["coverage"] - 0.95):
        print(f"-> EWMA-GK closer to 0.95 coverage by "
              f"{abs(b['coverage']-0.95) - abs(a['coverage']-0.95):.4f}")
    else:
        print(f"-> GARCH closer to 0.95 coverage by "
              f"{abs(a['coverage']-0.95) - abs(b['coverage']-0.95):.4f}")

    if a["mean_winkler"] < b["mean_winkler"]:
        print(f"-> EWMA-GK has lower mean Winkler by "
              f"{b['mean_winkler'] - a['mean_winkler']:,.2f}")
    else:
        print(f"-> GARCH has lower mean Winkler by "
              f"{a['mean_winkler'] - b['mean_winkler']:,.2f}")
