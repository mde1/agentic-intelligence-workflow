import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DB_PATH = "db/telegram_channels.db"
SOURCE_TABLE = "telegram_event_hourly"
OUTPUT_TABLE = "telegram_hourly_anomalies"


def load_hourly_data(conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
        SELECT
            hour_bucket,
            country,
            event_type,
            message_count,
            unique_channels
        FROM {SOURCE_TABLE}
    """
    return pd.read_sql_query(query, conn)


def build_hourly_anomalies(
    hourly: pd.DataFrame,
    baseline_hours: int = 168,   # 7 days
    min_baseline_points: int = 24,
    min_current_messages: int = 3,
    min_spike_ratio: float = 2.0,
    min_zscore: float = 2.5,
) -> pd.DataFrame:
    output_columns = [
        "hour_bucket",
        "country",
        "event_type",
        "current_message_count",
        "current_unique_channels",
        "baseline_avg_messages",
        "baseline_std_messages",
        "baseline_avg_channels",
        "baseline_std_channels",
        "baseline_points",
        "spike_ratio",
        "channel_spike_ratio",
        "zscore_messages",
        "zscore_channels",
        "is_anomaly",
        "anomaly_reason",
        "computed_at",
    ]

    if hourly.empty:
        return pd.DataFrame(columns=output_columns)

    hourly = hourly.copy()
    hourly["hour_bucket"] = pd.to_datetime(hourly["hour_bucket"], utc=True, errors="coerce")
    hourly = hourly.dropna(subset=["hour_bucket", "country", "event_type"])

    if hourly.empty:
        return pd.DataFrame(columns=output_columns)

    latest_hour = hourly["hour_bucket"].max()

    current = hourly[hourly["hour_bucket"] == latest_hour].copy()

    baseline_start = latest_hour - pd.Timedelta(hours=baseline_hours)
    baseline = hourly[
        (hourly["hour_bucket"] < latest_hour) &
        (hourly["hour_bucket"] >= baseline_start)
    ].copy()

    if baseline.empty or current.empty:
        return pd.DataFrame(columns=output_columns)

    baseline_stats = (
        baseline.groupby(["country", "event_type"], as_index=False)
        .agg(
            baseline_avg_messages=("message_count", "mean"),
            baseline_std_messages=("message_count", "std"),
            baseline_avg_channels=("unique_channels", "mean"),
            baseline_std_channels=("unique_channels", "std"),
            baseline_points=("message_count", "count"),
        )
    )

    merged = current.merge(
        baseline_stats,
        on=["country", "event_type"],
        how="left"
    )

    merged["baseline_std_messages"] = merged["baseline_std_messages"].fillna(0.0)
    merged["baseline_std_channels"] = merged["baseline_std_channels"].fillna(0.0)

    # avoid divide-by-zero for spike ratio
    merged["spike_ratio"] = np.where(
        merged["baseline_avg_messages"] > 0,
        merged["message_count"] / merged["baseline_avg_messages"],
        np.nan
    )

    merged["channel_spike_ratio"] = np.where(
        merged["baseline_avg_channels"] > 0,
        merged["unique_channels"] / merged["baseline_avg_channels"],
        np.nan
    )

    # z-scores
    merged["zscore_messages"] = np.where(
        merged["baseline_std_messages"] > 0,
        (merged["message_count"] - merged["baseline_avg_messages"]) / merged["baseline_std_messages"],
        np.nan
    )

    merged["zscore_channels"] = np.where(
        merged["baseline_std_channels"] > 0,
        (merged["unique_channels"] - merged["baseline_avg_channels"]) / merged["baseline_std_channels"],
        np.nan
    )

    # anomaly logic
    enough_history = merged["baseline_points"].fillna(0) >= min_baseline_points
    enough_volume = merged["message_count"] >= min_current_messages
    spike_condition = merged["spike_ratio"].fillna(0) >= min_spike_ratio
    zscore_condition = merged["zscore_messages"].fillna(0) >= min_zscore

    merged["is_anomaly"] = (
        enough_history &
        enough_volume &
        (spike_condition | zscore_condition)
    ).astype(int)

    def determine_reason(row) -> str | None:
        if row["is_anomaly"] != 1:
            return None

        reasons = []
        if pd.notna(row["spike_ratio"]) and row["spike_ratio"] >= min_spike_ratio:
            reasons.append(f"message spike {row['spike_ratio']:.1f}x baseline")
        if pd.notna(row["zscore_messages"]) and row["zscore_messages"] >= min_zscore:
            reasons.append(f"message z-score {row['zscore_messages']:.1f}")
        if pd.notna(row["channel_spike_ratio"]) and row["channel_spike_ratio"] >= min_spike_ratio:
            reasons.append(f"channel spike {row['channel_spike_ratio']:.1f}x baseline")

        return "; ".join(reasons) if reasons else "activity anomaly"

    merged["anomaly_reason"] = merged.apply(determine_reason, axis=1)

    merged["computed_at"] = datetime.now(timezone.utc).isoformat()

    result = merged.rename(columns={
        "message_count": "current_message_count",
        "unique_channels": "current_unique_channels",
    })

    result = result[output_columns].sort_values(
        ["is_anomaly", "spike_ratio", "zscore_messages", "current_message_count"],
        ascending=[False, False, False, False]
    )

    return result


def ensure_output_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    cursor.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} (
            hour_bucket TEXT,
            country TEXT,
            event_type TEXT,
            current_message_count INTEGER,
            current_unique_channels INTEGER,
            baseline_avg_messages REAL,
            baseline_std_messages REAL,
            baseline_avg_channels REAL,
            baseline_std_channels REAL,
            baseline_points INTEGER,
            spike_ratio REAL,
            channel_spike_ratio REAL,
            zscore_messages REAL,
            zscore_channels REAL,
            is_anomaly INTEGER,
            anomaly_reason TEXT,
            computed_at TEXT
        )
    """)
    conn.commit()


def save_hourly_anomalies(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if not df.empty:
        df.to_sql(OUTPUT_TABLE, conn, if_exists="append", index=False)

    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_hour
        ON {OUTPUT_TABLE}(hour_bucket)
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_flag
        ON {OUTPUT_TABLE}(is_anomaly)
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_country_event
        ON {OUTPUT_TABLE}(country, event_type)
    """)
    conn.commit()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        hourly = load_hourly_data(conn)
        anomalies = build_hourly_anomalies(
            hourly,
            baseline_hours=168,      # 7 days
            min_baseline_points=24,
            min_current_messages=3,
            min_spike_ratio=2.0,
            min_zscore=2.5,
        )
        ensure_output_table(conn)
        save_hourly_anomalies(conn, anomalies)
        flagged = int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0
        print(f"Wrote {len(anomalies)} rows to {OUTPUT_TABLE}")
        print(f"Flagged {flagged} anomalies for latest hour")
    finally:
        conn.close()


if __name__ == "__main__":
    main()