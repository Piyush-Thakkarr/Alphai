from __future__ import annotations

"""
Walk-forward backtest of the predictor over the last 720 hourly bars.

For each bar N in the test set:
    1. Slice ohlc to bars [..N-1] (NEVER include bar N)
    2. Take the last `WINDOW` bars (or fewer if not yet available)
    3. Call predict_range -> 95% interval for bar N's close
    4. Compare to actual close of bar N
    5. Record (low, high, actual, hit, width, winkler)

At the end, write all predictions to backtest_results.jsonl and print
the three headline metrics: coverage_95, mean_width_95, mean_winkler_95.

The brief specifies "look only at data up to that bar" - this is the
no-look-ahead discipline. The slice ohlc.iloc[..N] in Python excludes
index N, which guarantees bar N's price never enters its own prediction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import fetch_klines
from model import predict_range

# Match the dashboard's window size so backtest reflects production
WINDOW = 500
# Brief: "For each of those 720 bars, pretend you don't know the future."
# So we want 720 predictions in the reported metrics. We fetch a small
# warmup buffer of bars BEFORE the 720-bar test set so the model has
# history for its very first prediction. Reported metrics are over
# exactly 720 predictions.
TEST_BARS = 720
WARMUP_BARS = 10
ALPHA = 0.05  # 95% interval


def winkler_score(low: float, high: float, actual: float, alpha: float = ALPHA) -> float:
    """
    Winkler interval score (lower is better).

    If actual is inside [low, high]:    score = width
    If actual < low:                    score = width + (2/alpha)*(low - actual)
    If actual > high:                   score = width + (2/alpha)*(actual - high)

    Penalizes both wide intervals AND misses, with the miss penalty
    scaling by 1/alpha so a 95% interval miss is severely punished.
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
    Walk forward producing one prediction for each of the LAST `test_bars`.

    `ohlc` should contain at least test_bars + warmup bars. The first
    portion is treated as warmup (model history only, not predicted on);
    we predict exactly the most recent `test_bars` bars of the input.

    For the bar at index N in the test set, we slice ohlc.iloc[:N] - which
    EXCLUDES row N. That guarantees no look-ahead: bar N's price is
    structurally impossible to enter its own prediction.
    """
    rng = np.random.default_rng(42)
    results = []

    n_bars = len(ohlc)
    if n_bars < test_bars + 1:
        print(
            f"  WARNING: only {n_bars} bars available, less than the requested "
            f"{test_bars} test bars + 1 warmup. Producing {max(0, n_bars - 1)} "
            f"predictions instead of {test_bars}."
        )
    test_start = max(1, n_bars - test_bars)  # index of first test bar

    for n in tqdm(range(test_start, n_bars), desc="Backtest"):
        # Information set at prediction time: bars [max(0, n-window), n).
        # Never includes bar n.
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
            "current_price": prediction["current_price"],   # close at bar n-1
            "actual_close": actual_close,                    # close at bar n
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
    # Fetch test set (720 bars) + small warmup buffer + 1 to compensate
    # for the in-progress current bar that fetch_klines filters out.
    fetch_count = TEST_BARS + WARMUP_BARS + 1
    print(f"Fetching {fetch_count} hourly BTCUSDT bars from Binance...")
    ohlc = fetch_klines(limit=fetch_count)
    print(f"Got {len(ohlc)} closed bars from {ohlc.index[0]} to {ohlc.index[-1]}")
    print(f"Test set: most recent {TEST_BARS} bars; warmup: the {len(ohlc) - TEST_BARS} earlier bars\n")

    results = run_backtest(ohlc)
    summary = summarize(results)

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Predictions:        {summary['n_predictions']}")
    print(f"Coverage @ 95%:     {summary['coverage_95']:.4f}  (target: 0.9500)")
    print(f"Mean width:         ${summary['mean_width_95']:,.2f}")
    print(f"Median width:       ${summary['median_width_95']:,.2f}")
    print(f"Mean Winkler:       {summary['mean_winkler_95']:,.2f}  (lower is better)")
    print(f"Median Winkler:     {summary['median_winkler_95']:,.2f}")
    print("=" * 50)

    out_path = Path(__file__).parent / "backtest_results.jsonl"
    save_jsonl(results, out_path)
    print(f"\nWrote {len(results)} predictions to {out_path}")
