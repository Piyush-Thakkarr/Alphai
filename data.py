"""
Fetch BTCUSDT 1-hour OHLC bars from Binance's public data mirror.

Why the mirror (data-api.binance.vision) and not api.binance.com:
    api.binance.com is geo-blocked in some regions (incl. India). The
    data-api.binance.vision mirror serves the same public market data
    with no geo-block and no authentication.

Why we keep OHLC (not just close):
    Garman-Klass volatility uses Open, High, Low, Close per bar.
    It is ~7x more statistically efficient than close-to-close stdev,
    which directly tightens our 95% prediction interval.
"""

import requests
import pandas as pd

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 720) -> pd.DataFrame:
    """
    Pull the most recent `limit` candlesticks from Binance.

    Binance returns at most 1000 bars per request. For our project
    we need 720 (last 30 days of hourly) for the backtest, and 500
    for the dashboard window. Both fit in a single request.

    Returns a DataFrame indexed by close_time (UTC) with columns:
        open_time, open, high, low, close, volume.

    The "close_time" index means: the row at index T is the bar that
    closed at exactly T. This makes the no-peeking discipline explicit:
    when predicting bar T+1, only rows with index <= T are allowed.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()

    # Validate the response is a non-empty list of klines. Binance error
    # payloads come back as a dict like {"code": -1121, "msg": "Invalid symbol."}
    # which would otherwise silently produce an empty DataFrame.
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"Unexpected Binance response: {raw!r}. "
            f"Expected a non-empty list of kline arrays."
        )

    # Binance kline schema:
    # [open_time, open, high, low, close, volume, close_time, quote_volume,
    #  trades, taker_buy_base, taker_buy_quote, ignore]
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"
    ])

    # Cast prices to float (Binance returns them as strings)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Convert millisecond timestamps to UTC datetimes
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Drop the in-progress bar if present. Binance includes the current
    # (not-yet-closed) hour at the tail. We never use unclosed bars.
    # pd.Timestamp.now(tz="UTC") is the documented stable form;
    # pd.Timestamp.utcnow() varies in tz-awareness across pandas versions.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now_utc]

    return df.set_index("close_time")[["open_time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    # Sanity check: pull 720 bars and confirm the shape and timing.
    df = fetch_klines(limit=720)
    print(df.tail(3))
    print()
    print(f"Rows fetched:   {len(df)}")
    print(f"First bar close: {df.index[0]}")
    print(f"Last bar close:  {df.index[-1]}")
    span_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    print(f"Span:            {span_hours:.1f} hours ({span_hours/24:.1f} days)")
