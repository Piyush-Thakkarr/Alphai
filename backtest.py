from __future__ import annotations

"""
Walk-forward backtest over the most recent 720 hourly bars.

For each bar N in the test set, slice ohlc.iloc[start:n] (half-open, so
N itself is excluded) and call predict_range. Compare the predicted
[low, high] to the actual close at bar N. Record everything.

We fetch a small warmup buffer before the 720-bar test window so the
model has history for the very first prediction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import fetch_klines
from model import predict_range

WINDOW = 500              # rolling history window for sigma + df fit
TEST_BARS = 720           # the actual test set (the brief's 30 days)
WARMUP_BARS = 10          # extra bars before the test set
ALPHA = 0.05              # 95% interval


def winkler_score(low: float, high: float, actual: float, alpha: float = ALPHA) -> float:
    """
    Lower is better. Inside [low, high]: score = width.
    Outside: width + (2/alpha) * distance to nearest bound.
    The 2/alpha penalty makes a 95% miss expensive.
    """
    width = high - low
    if actual < low:
        return width + (2.0 / alpha) * (low - actual)
    if actual > high:
        return width + (2.0 / alpha) * (actual - high)
    return width


def run_backtest(ohlc: pd.DataFrame, window: int = WINDOW, alpha: float = ALPHA,
                 test_bars: int = TEST_BARS) -> list[dict]:
    """
    Walk forward, one prediction per bar in the test set.
    The slice ohlc.iloc[start:n] excludes index n, so bar n's price
    can't enter its own prediction.
    """
    rng = np.random.default_rng(42)
    results = []

    n_bars = len(ohlc)
    if n_bars < test_bars + 1:
        print(f"  only {n_bars} bars available, want {test_bars + 1}. "
              f"producing {max(0, n_bars - 1)} predictions.")
    test_start = max(1, n_bars - test_bars)

    for n in tqdm(range(test_start, n_bars), desc="backtest"):
        start = max(0, n - window)
        info_set = ohlc.iloc[start:n]

        prediction = predict_range(info_set, alpha=alpha, rng=rng)

        actual_close = float(ohlc["close"].iloc[n])
        bar_close_time = ohlc.index[n]

        in_range = prediction["low"] <= actual_close <= prediction["high"]
        width = prediction["high"] - prediction["low"]
        wink = winkler_score(prediction["low"], prediction["high"], actual_close, alpha)

        results.append({
            "bar_close_time": bar_close_time.isoformat(),
            "current_price": prediction["current_price"],
            "actual_close": actual_close,
            "low_95": prediction["low"],
            "high_95": prediction["high"],
            "width_95": width,
            "in_range": int(in_range),
            "winkler_95": wink,
            "sigma_hourly": prediction["sigma"],
            "df_t": prediction["df_t"],
        })

    return results


def summarize(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    return {
        "n_predictions": len(df),
        "coverage_95": float(df["in_range"].mean()),
        "mean_width_95": float(df["width_95"].mean()),
        "median_width_95": float(df["width_95"].median()),
        "mean_winkler_95": float(df["winkler_95"].mean()),
        "median_winkler_95": float(df["winkler_95"].median()),
    }


def save_jsonl(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    with path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    fetch_count = TEST_BARS + WARMUP_BARS + 1   # +1 because the in-progress bar gets filtered
    print(f"fetching {fetch_count} bars from binance...")
    ohlc = fetch_klines(limit=fetch_count)
    print(f"got {len(ohlc)} closed bars from {ohlc.index[0]} to {ohlc.index[-1]}")
    print(f"test set: most recent {TEST_BARS}, warmup: {len(ohlc) - TEST_BARS}\n")

    results = run_backtest(ohlc)
    s = summarize(results)

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"predictions:    {s['n_predictions']}")
    print(f"coverage @ 95%: {s['coverage_95']:.4f}  (target 0.9500)")
    print(f"mean width:     ${s['mean_width_95']:,.2f}")
    print(f"median width:   ${s['median_width_95']:,.2f}")
    print(f"mean winkler:   {s['mean_winkler_95']:,.2f}")
    print(f"median winkler: {s['median_winkler_95']:,.2f}")
    print("=" * 50)

    out = Path(__file__).parent / "backtest_results.jsonl"
    save_jsonl(results, out)
    print(f"\nwrote {len(results)} predictions to {out}")
