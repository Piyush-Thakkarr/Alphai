"""
streamlit dashboard. on each visit:
fetch latest 500 bars, run predict_range, show current price + 95%
range + chart + sample histogram. saves prediction to sqlite for the
live history section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data import fetch_klines
from model import predict_range
from persistence import init_db, save_prediction, update_actuals, load_history

DASHBOARD_BARS = 500   # brief says use last 500 bars
CHART_BARS = 50        # brief says show last 50
BACKTEST_PATH = Path(__file__).parent / "backtest_results.jsonl"


@st.cache_data(ttl=60)
def load_recent_bars(n: int) -> pd.DataFrame:
    # cache 60s so refreshes don't spam binance
    return fetch_klines(limit=n)


@st.cache_resource
def db_initialized() -> bool:
    # run init_db once per session, not every render
    init_db()
    return True


def load_backtest_metrics() -> dict | None:
    if not BACKTEST_PATH.exists():
        return None
    rows = [json.loads(line) for line in BACKTEST_PATH.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "n": len(df),
        "coverage_95": float(df["in_range"].mean()),
        "mean_width_95": float(df["width_95"].mean()),
        "mean_winkler_95": float(df["winkler_95"].mean()),
    }


def regime_tag(sigma: float) -> tuple[str, str]:
    # cutoffs picked from typical btc hourly sigma ranges
    if sigma < 0.003:
        return "Calm", "#1f77b4"
    if sigma < 0.007:
        return "Moderate", "#ff7f0e"
    return "Chaotic", "#d62728"


st.set_page_config(page_title="BTC Next-Hour Forecast", layout="wide")
st.title("BTC Next-Hour 95% Forecast")
st.caption(
    "GBM, garman-klass + EWMA volatility, student-t innovations, "
    "10k MC. backtested walk-forward on the last 30 days."
)

# Backtest headline metrics row
metrics = load_backtest_metrics()
if metrics is None:
    st.warning("No backtest_results.jsonl found. Run `python backtest.py` first.")
else:
    cols = st.columns(4)
    cols[0].metric("Backtest predictions", f"{metrics['n']}")
    cols[1].metric("Coverage @ 95%", f"{metrics['coverage_95']:.4f}", help="Target: 0.95")
    cols[2].metric("Mean width", f"${metrics['mean_width_95']:,.0f}")
    cols[3].metric("Mean Winkler", f"{metrics['mean_winkler_95']:,.0f}", help="Lower is better")

st.divider()

try:
    with st.spinner("fetching latest bars and running model..."):
        bars = load_recent_bars(DASHBOARD_BARS)
        # seed rng from latest bar timestamp so refreshes in the same
        # hour give the same prediction (only changes when a new bar closes)
        rng_seed = int(bars.index[-1].value & 0xFFFFFFFF)
        result = predict_range(bars, rng=np.random.default_rng(rng_seed))
except Exception as exc:
    st.error(f"couldn't fetch from binance or run the model. "
             f"likely a network blip, try refreshing in a few seconds.\n\n{exc}")
    st.stop()

current_price = result["current_price"]
low = result["low"]
high = result["high"]
width = high - low
sigma = result["sigma"]
df_t = result["df_t"]
samples = result["samples"]
regime, regime_color = regime_tag(sigma)
last_close_time = bars.index[-1]

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Forecast for the hour after {last_close_time.strftime('%Y-%m-%d %H:%M UTC')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current BTC price", f"${current_price:,.2f}")
    c2.metric("95% low", f"${low:,.2f}", f"{(low/current_price - 1)*100:+.2f}%")
    c3.metric("95% high", f"${high:,.2f}", f"{(high/current_price - 1)*100:+.2f}%")
    st.write(f"Range width: **${width:,.2f}** ({width/current_price*100:.2f}% of price)")

with right:
    st.subheader("Volatility regime")
    st.markdown(f"<h2 style='color:{regime_color};margin:0;'>{regime}</h2>", unsafe_allow_html=True)
    st.write(f"Hourly sigma: **{sigma*100:.3f}%**")
    st.write(f"Student-t df: **{df_t:.2f}**")

st.divider()

# last 50 bars + ribbon for next-hour range
chart_data = bars.tail(CHART_BARS)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(chart_data.index, chart_data["close"], color="#1f77b4", linewidth=2, label="Close")

next_bar_time = last_close_time + pd.Timedelta(hours=1)
ribbon_x = [last_close_time, next_bar_time]
ax.fill_between(
    ribbon_x, [low, low], [high, high],
    color="orange", alpha=0.30, edgecolor="orange", linewidth=1.2, label="Next-hour 95% range"
)
# dotted line at current price = "what if price doesn't move"
ax.plot(
    [last_close_time, next_bar_time], [current_price, current_price],
    color="black", linewidth=1.0, linestyle=":", label="No-change reference",
)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.set_xlabel("UTC time")
ax.set_ylabel("BTC / USDT")
ax.set_title(f"Last {CHART_BARS} bars + next-hour 95% prediction")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.autofmt_xdate()
st.pyplot(fig)

# histogram of the 10k simulated next-hour prices
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.hist(samples, bins=80, color="#1f77b4", alpha=0.7, edgecolor="white")
ax2.axvline(low, color="red", linestyle="--", linewidth=1.2, label=f"low ${low:,.0f}")
ax2.axvline(high, color="red", linestyle="--", linewidth=1.2, label=f"high ${high:,.0f}")
ax2.axvline(current_price, color="black", linewidth=1.5, label=f"current ${current_price:,.0f}")
ax2.set_xlabel("Simulated next-hour close price")
ax2.set_ylabel("count")
ax2.set_title(f"Distribution of {len(samples):,} simulated next-hour prices")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)

st.caption(
    "data: binance public mirror (data-api.binance.vision). "
    "bars cached 60s. "
    f"latest closed bar: {last_close_time.strftime('%Y-%m-%d %H:%M UTC')}."
)

# part C: persistence + live history
st.divider()
st.subheader("Live prediction history")

db_initialized()

# INSERT OR IGNORE on predicted_for_unix means refreshes in the same hour
# don't overwrite the first stored prediction
predicted_for_time = last_close_time + pd.Timedelta(hours=1)
made_at = pd.Timestamp.now(tz="UTC")
save_prediction(
    {
        "current_price": current_price,
        "low": low,
        "high": high,
        "sigma": sigma,
        "df_t": df_t,
    },
    last_close_time=last_close_time,
    predicted_for_time=predicted_for_time,
    made_at=made_at,
)

# for any past prediction whose target hour has closed, fill in actual close
update_actuals(bars)

history = load_history()

if history.empty:
    st.info("No predictions stored yet. This is the first visit.")
else:
    resolved = history.dropna(subset=["actual_close"])
    n_pending = int(history["actual_close"].isna().sum())

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Predictions stored", f"{len(history)}")
    h2.metric("Resolved", f"{len(resolved)}")
    h3.metric("Pending (waiting on bar close)", f"{n_pending}")
    if len(resolved) > 0:
        live_cov = float(resolved["in_range"].mean())
        h4.metric("Live coverage", f"{live_cov:.4f}", help="in_range / resolved")
    else:
        h4.metric("Live coverage", "n/a")

    # timeline: blue bars are the stored 95% ranges, green dots = hits, red x = misses
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for _, row in history.iterrows():
        x = row["predicted_for_time"]
        ax3.plot([x, x], [row["low_95"], row["high_95"]], color="#1f77b4", alpha=0.55, linewidth=1.5)
    if not resolved.empty:
        hits = resolved[resolved["in_range"] == 1]
        miss = resolved[resolved["in_range"] == 0]
        if not hits.empty:
            ax3.scatter(hits["predicted_for_time"], hits["actual_close"],
                        color="#2ca02c", zorder=5, label=f"hits ({len(hits)})", s=24)
        if not miss.empty:
            ax3.scatter(miss["predicted_for_time"], miss["actual_close"],
                        color="#d62728", zorder=5, marker="x", label=f"misses ({len(miss)})", s=40)
    ax3.set_xlabel("Predicted hour (UTC)")
    ax3.set_ylabel("BTC / USDT")
    ax3.set_title("Stored predictions: 95% range (blue bars) and realized close (dots)")
    ax3.grid(alpha=0.3)
    if not resolved.empty:
        ax3.legend(loc="upper left")
    fig3.autofmt_xdate()
    st.pyplot(fig3)

    with st.expander("Raw prediction history (latest 50)"):
        cols_to_show = [
            "predicted_for_time", "current_price", "low_95", "high_95",
            "actual_close", "in_range", "sigma", "df_t",
        ]
        st.dataframe(history[cols_to_show].tail(50), use_container_width=True)
