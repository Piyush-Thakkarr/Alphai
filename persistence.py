"""
SQLite persistence for the live dashboard (Part C).

Goal: every visit saves the prediction it just made; over time the
database grows into a record of predictions with their actuals filled
in once the predicted hour has elapsed.

Schema (single table, time keyed by Unix seconds for tz-safety):

    predictions(
        predicted_for_unix    INTEGER PRIMARY KEY,
        prediction_made_unix  INTEGER NOT NULL,
        last_close_unix       INTEGER NOT NULL,
        current_price         REAL    NOT NULL,
        low_95                REAL    NOT NULL,
        high_95               REAL    NOT NULL,
        sigma                 REAL,
        df_t                  REAL,
        actual_close          REAL,
        in_range              INTEGER
    )

INSERT OR IGNORE on the primary key means: the FIRST prediction made
for a given hour is the one we keep. Repeat visits in the same hour
don't overwrite it - which is what you want for evaluation purity.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    predicted_for_unix   INTEGER PRIMARY KEY,
    prediction_made_unix INTEGER NOT NULL,
    last_close_unix      INTEGER NOT NULL,
    current_price        REAL    NOT NULL,
    low_95               REAL    NOT NULL,
    high_95              REAL    NOT NULL,
    sigma                REAL,
    df_t                 REAL,
    actual_close         REAL,
    in_range             INTEGER
);
"""


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _to_unix(ts: pd.Timestamp) -> int:
    """pd.Timestamp -> Unix seconds (handles both tz-aware and naive)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def init_db() -> None:
    """Create the predictions table if not present. Safe to call repeatedly."""
    with _conn() as c:
        c.executescript(SCHEMA)


def save_prediction(
    prediction: dict,
    last_close_time: pd.Timestamp,
    predicted_for_time: pd.Timestamp,
    made_at: pd.Timestamp,
) -> bool:
    """
    Persist one prediction. Returns True if newly inserted, False if a
    prediction for this hour was already recorded (then it's skipped).
    """
    with _conn() as c:
        cur = c.execute(
            """
            INSERT OR IGNORE INTO predictions
                (predicted_for_unix, prediction_made_unix, last_close_unix,
                 current_price, low_95, high_95, sigma, df_t)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _to_unix(predicted_for_time),
                _to_unix(made_at),
                _to_unix(last_close_time),
                float(prediction["current_price"]),
                float(prediction["low"]),
                float(prediction["high"]),
                float(prediction["sigma"]),
                float(prediction["df_t"]),
            ),
        )
        return cur.rowcount == 1


def update_actuals(bars: pd.DataFrame) -> int:
    """
    For any prediction whose target hour has elapsed, fill in the actual
    close and the hit indicator from `bars`. Returns the number of rows
    updated.

    `bars` should be the current pull from Binance, indexed by close_time.
    """
    if bars.empty:
        return 0

    # Build a map: unix -> close, for fast lookup
    bars = bars.copy()
    bars["unix"] = bars.index.map(_to_unix)
    close_by_unix = dict(zip(bars["unix"], bars["close"]))

    updated = 0
    with _conn() as c:
        rows = c.execute(
            """
            SELECT predicted_for_unix, low_95, high_95
            FROM predictions
            WHERE actual_close IS NULL
            """
        ).fetchall()

        for predicted_for_unix, low, high in rows:
            actual = close_by_unix.get(int(predicted_for_unix))
            if actual is None:
                continue
            in_range = int(low <= actual <= high)
            c.execute(
                """
                UPDATE predictions
                SET actual_close = ?, in_range = ?
                WHERE predicted_for_unix = ?
                """,
                (float(actual), in_range, int(predicted_for_unix)),
            )
            updated += 1

    return updated


def load_history() -> pd.DataFrame:
    """
    Load all stored predictions as a DataFrame, ordered by target time.
    Adds tz-aware Timestamp columns for the three time fields.
    """
    with _conn() as c:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY predicted_for_unix",
            c,
        )

    if df.empty:
        return df

    for src, dst in [
        ("predicted_for_unix", "predicted_for_time"),
        ("prediction_made_unix", "prediction_made_at"),
        ("last_close_unix", "last_close_time"),
    ]:
        df[dst] = pd.to_datetime(df[src], unit="s", utc=True)

    return df
