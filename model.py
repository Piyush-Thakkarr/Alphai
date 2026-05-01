"""
GBM forecaster for next-hour BTC price range.

Pipeline:
    1. Garman-Klass per-bar variance from OHLC
    2. EWMA-smooth across bars to capture volatility clustering
    3. Fit Student-t degrees-of-freedom from log returns (fat tails)
    4. Monte Carlo: simulate 10,000 next-hour log returns under GBM, mu=0
    5. Read 2.5th and 97.5th percentiles -> 95% interval

This is the SINGLE source of truth for predictions. backtest.py and
app.py both call predict_range() so live and historical numbers agree
by construction (no train/serve skew).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

EWMA_LAMBDA = 0.97   # half-life ~ 23 bars at lambda=0.97
DF_FLOOR = 4.0       # df>4 keeps kurtosis finite (numerical safety)
N_SIMS = 10_000      # Monte Carlo paths per prediction


def garman_klass_per_bar(ohlc: pd.DataFrame) -> pd.Series:
    """
    Per-bar variance estimator using OHLC.

        sigma2_GK = 0.5 * [ln(H/L)]^2 - (2*ln(2) - 1) * [ln(C/O)]^2

    ~7x more efficient than close-to-close stdev. Uses all four OHLC
    values per bar to estimate within-bar variance. Tighter sigma ->
    tighter 95% interval -> lower Winkler score.
    """
    log_hl = np.log(ohlc["high"] / ohlc["low"])
    log_co = np.log(ohlc["close"] / ohlc["open"])
    return 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2


def ewma_sigma(gk_series: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """
    EWMA across the per-bar GK variance series. Returns sigma at the
    most recent bar.

        sigma2_t = (1 - lambda) * GK_{t-1} + lambda * sigma2_{t-1}

    Recency-weighting captures volatility clustering: recent shocks
    influence current sigma more than distant ones. lambda=0.97 gives
    a half-life of ~23 bars (~1 day for hourly BTC).
    """
    smoothed = gk_series.ewm(alpha=1.0 - lam, adjust=False).mean()
    return float(np.sqrt(smoothed.iloc[-1]))


def fit_student_t_df(prices: pd.Series, floor: float = DF_FLOOR) -> float:
    """
    Fit Student-t degrees of freedom on standardized log returns of
    the prices in this window.

    df controls tail thickness. Lower df -> fatter tails. df -> infty
    is the Normal distribution. Floor at 4 keeps kurtosis finite.

    Fits on the same window passed to predict_range, so no look-ahead:
    when called at bar N with ohlc[..N-1], df sees only past returns.
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
    Predict the (1 - alpha) interval for the next bar's close.

    Uses ONLY the rows in `ohlc`. The caller is responsible for slicing
    to avoid look-ahead. At backtest bar N, pass ohlc.iloc[:N].

    Args:
        ohlc:   DataFrame with columns open/high/low/close, chronological.
                Last row = most recent closed bar.
        alpha:  tail probability. 0.05 -> 95% interval.
        n_sims: Monte Carlo sample size.
        rng:    optional numpy Generator for reproducibility.

    Returns dict with keys:
        low, high       interval bounds in price units
        sigma           hourly volatility used (log-return units)
        df_t            Student-t df fitted on this window
        samples         np.ndarray of n_sims simulated next-hour prices
        current_price   last close in input
    """
    if rng is None:
        rng = np.random.default_rng()

    # Defensive: Student-t df fitting on tiny samples is unstable and can
    # hang the optimizer. 10 bars (= 9 returns) is the safe floor.
    if len(ohlc) < 10:
        raise ValueError(
            f"predict_range needs at least 10 bars, got {len(ohlc)}. "
            f"Pass a larger window."
        )
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc is missing required columns: {sorted(missing)}")

    # 1. Per-bar GK variance -> EWMA smooth -> current sigma
    gk = garman_klass_per_bar(ohlc)
    sigma = ewma_sigma(gk)

    # 2. Student-t df from this window's log returns (no look-ahead)
    df_t = fit_student_t_df(ohlc["close"])

    # 3. Monte Carlo: simulate next-hour log returns under GBM, mu=0
    #
    #    Under GBM with zero drift:
    #        log(S_{t+1}/S_t) = -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z
    #    Here dt = 1 (one hour). The -0.5*sigma^2 is the Ito correction:
    #    it ensures E[S_{t+1}] = S_t even though log returns have a
    #    slightly negative mean (because exp is convex).
    #
    #    Z is standardized Student-t with unit variance. Raw standard_t
    #    has variance df/(df-2); we rescale by sqrt((df-2)/df) so that
    #    sigma * Z has variance sigma^2, matching our GK+EWMA estimate.
    raw_t = rng.standard_t(df_t, size=n_sims)
    z = raw_t * np.sqrt((df_t - 2.0) / df_t)
    log_returns = -0.5 * sigma**2 + sigma * z

    current_price = float(ohlc["close"].iloc[-1])
    samples = current_price * np.exp(log_returns)

    # 4. Empirical (1 - alpha) interval from MC samples
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
    # Sanity check on live Binance data
    from data import fetch_klines

    df = fetch_klines(limit=720)
    result = predict_range(df, rng=np.random.default_rng(42))

    width = result["high"] - result["low"]
    width_pct = width / result["current_price"] * 100

    print(f"Current price:   ${result['current_price']:,.2f}")
    print(f"95% range:       ${result['low']:,.2f}  ->  ${result['high']:,.2f}")
    print(f"  width:         ${width:,.2f}  ({width_pct:.2f}%)")
    print(f"sigma (hourly):  {result['sigma']:.5f}  (~{result['sigma']*100:.3f}% per hour)")
    print(f"Student-t df:    {result['df_t']:.2f}")
