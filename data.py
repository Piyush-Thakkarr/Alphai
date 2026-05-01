"""
Pull 1h OHLC bars from binance public mirror (api.binance.com is
geo-blocked in india, the data-api.binance.vision mirror isn't).
We keep all OHLC because garman-klass needs O/H/L/C per bar.
"""

import requests
import pandas as pd

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 720) -> pd.DataFrame:
    """
    Pull `limit` most recent klines. binance caps at 1000 per request.
    Returns a df indexed by close_time (UTC) with O/H/L/C/V.
    The in-progress current bar gets filtered out so we only have
    closed bars.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()

    # binance returns a dict with "code"/"msg" on errors instead of klines
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"unexpected binance response: {raw!r}")

    # schema: [open_time, o, h, l, c, v, close_time, qav, trades,
    #         tb_base, tb_quote, ignore]
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"
    ])

    # binance sends prices as strings
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # drop any bar whose close hasn't actually happened yet
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now_utc]

    return df.set_index("close_time")[["open_time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = fetch_klines(limit=720)
    print(df.tail(3))
    print(f"\nrows: {len(df)}")
    print(f"first close: {df.index[0]}")
    print(f"last close:  {df.index[-1]}")
    span_h = (df.index[-1] - df.index[0]).total_seconds() / 3600
    print(f"span: {span_h:.1f} hours ({span_h/24:.1f} days)")
