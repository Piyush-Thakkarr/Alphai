"""
Fetch sub-hour bars (down to 1m) from binance for realized-variance
estimation. Paginated since binance caps at 1000 bars per request.
"""

from __future__ import annotations
import time
import requests
import pandas as pd

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def _interval_ms(interval: str) -> int:
    """Convert binance interval string to milliseconds."""
    n = int(interval[:-1])
    unit = interval[-1]
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def fetch_klines_paginated(symbol: str, interval: str, total_bars: int) -> pd.DataFrame:
    """
    Pull `total_bars` of `interval` bars ending at the most recent
    closed bar. Paginates by repeatedly walking endTime backward.
    """
    step_ms = _interval_ms(interval)
    chunk_size = 1000
    end_time_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    all_rows = []
    fetched = 0
    while fetched < total_bars:
        want = min(chunk_size, total_bars - fetched)
        params = {
            "symbol": symbol,
            "interval": interval,
            "endTime": end_time_ms,
            "limit": want,
        }
        r = requests.get(BINANCE_URL, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not isinstance(chunk, list) or not chunk:
            break
        all_rows = chunk + all_rows
        fetched += len(chunk)
        # walk endTime back to just before the earliest bar we just got
        end_time_ms = chunk[0][0] - 1
        # be polite to the public mirror
        time.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # drop the in-progress bar at the tail
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now_utc]

    # de-dup in case pagination overlapped
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return df.set_index("close_time")[["open_time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    # sanity: pull last hour of 1-min bars
    df = fetch_klines_paginated("BTCUSDT", "1m", 60)
    print(df.tail(3))
    print(f"\nrows: {len(df)}, span: {df.index[-1] - df.index[0]}")
