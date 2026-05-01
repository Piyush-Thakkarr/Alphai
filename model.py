"""
GBM forecaster for next-hour BTC close.

pipeline:
  - garman-klass per-bar variance from OHLC
  - EWMA across bars (lambda=0.97)
  - student-t df fitted on this window's log returns, floor 4
  - 10k MC paths under GBM with mu=0
  - 2.5 / 97.5 percentiles -> 95% interval

predict_range() is called by both backtest.py and app.py so they
always agree.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

EWMA_LAMBDA = 0.97   # half-life ~23 bars
DF_FLOOR = 4.0       # df>4 keeps kurtosis finite
N_SIMS = 10_000


def garman_klass_per_bar(ohlc: pd.DataFrame) -> pd.Series:
    """
    GK variance per bar from O/H/L/C:
      sigma2 = 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2
    ~7x more efficient than close-to-close stdev because it uses
    all four prices per bar.
    """
    log_hl = np.log(ohlc["high"] / ohlc["low"])
    log_co = np.log(ohlc["close"] / ohlc["open"])
    return 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2


def ewma_sigma(gk_series: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """
    EWMA-smooth GK variance series, return sigma at the latest bar.
    Recursion: sigma2_t = (1-lam)*GK_{t-1} + lam*sigma2_{t-1}.
    Recent shocks weight more. lam=0.97 -> half-life ~23 bars.
    """
    smoothed = gk_series.ewm(alpha=1.0 - lam, adjust=False).mean()
    return float(np.sqrt(smoothed.iloc[-1]))


def fit_student_t_df(prices: pd.Series, floor: float = DF_FLOOR) -> float:
    """
    Fit student-t df from log returns, with a floor at 4.
    Lower df = fatter tails. Floor avoids df<4 (undefined kurtosis).
    Standardization here is unconditional (uses sample stdev, not
    conditional sigma_t). Theoretically less rigorous than t-GARCH
    but coverage comes out at 0.95 anyway.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    standardized = (log_ret - log_ret.mean()) / log_ret.std(ddof=1)
    df_fit, _loc, _scale = stats.t.fit(standardized, floc=0.0, fscale=1.0)
    return max(float(df_fit), floor)


def predict_range(
    ohlc: pd.DataFrame,
    alpha: float = 0.05,
    n_sims: int = N_SIMS,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Return the (1-alpha) interval for the next bar's close.
    Caller is responsible for slicing ohlc to avoid look-ahead.
    For backtest at bar N, pass ohlc.iloc[:N] (excludes N).

    Returns dict: low, high, sigma, df_t, samples, current_price.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(ohlc) < 10:
        raise ValueError(f"need at least 10 bars, got {len(ohlc)}")
    missing = {"open", "high", "low", "close"} - set(ohlc.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")

    # per-bar GK -> EWMA -> sigma
    gk = garman_klass_per_bar(ohlc)
    sigma = ewma_sigma(gk)

    # fit df on this window's returns (no peeking — uses input only)
    df_t = fit_student_t_df(ohlc["close"])

    # GBM step under mu=0:
    #   log(S_t+1/S_t) = -0.5*sigma^2 + sigma*Z
    # the -0.5*sigma^2 is the ito correction. Z is unit-variance student-t
    # (raw standard_t has variance df/(df-2), rescaled to 1).
    raw_t = rng.standard_t(df_t, size=n_sims)
    z = raw_t * np.sqrt((df_t - 2.0) / df_t)
    log_returns = -0.5 * sigma**2 + sigma * z

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


if __name__ == "__main__":
    from data import fetch_klines

    df = fetch_klines(limit=720)
    result = predict_range(df, rng=np.random.default_rng(42))

    width = result["high"] - result["low"]
    width_pct = width / result["current_price"] * 100

    print(f"current price:   ${result['current_price']:,.2f}")
    print(f"95% range:       ${result['low']:,.2f} - ${result['high']:,.2f}")
    print(f"  width:         ${width:,.2f}  ({width_pct:.2f}%)")
    print(f"sigma (hourly):  {result['sigma']:.5f}")
    print(f"student-t df:    {result['df_t']:.2f}")
