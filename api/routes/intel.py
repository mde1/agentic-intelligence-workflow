from fastapi import APIRouter, Query
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "db"

TELEGRAM_DB = DB_DIR / "telegram_channels.db"
MARKETS_DB = DB_DIR / "market_data.db"
EARTHQUAKE_DB = DB_DIR / "earthquakes.db"  # change if your real filename differs


def db_exists(path: Path) -> bool:
    return path.exists()


def safe_query(db_path: Path, query: str, params=None) -> pd.DataFrame:
    """
    Execute a query safely against a SQLite database.
    Returns an empty DataFrame if the database, table, or query fails.
    """
    if not db_exists(db_path):
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn, params=params or [])
    except Exception:
        return pd.DataFrame()


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def clean_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    cleaned = df.copy()

    # Convert datetime-like columns safely
    for col in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype("object")

    records = cleaned.to_dict(orient="records")
    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in records
    ]


def get_latest_hour() -> str | None:
    latest_hour_df = safe_query(
        TELEGRAM_DB,
        """
        SELECT MAX(hour_bucket) AS latest_hour
        FROM telegram_hourly_anomalies
        """,
    )
    if latest_hour_df.empty:
        return None

    latest_hour = latest_hour_df.iloc[0].get("latest_hour")
    return str(latest_hour) if pd.notna(latest_hour) else None


def get_hourly_anomalies(latest_hour: str | None, anomaly_limit: int) -> pd.DataFrame:
    if anomaly_limit <= 0 or latest_hour is None:
        return pd.DataFrame()

    return safe_query(
        TELEGRAM_DB,
        """
        SELECT
            hour_bucket,
            country,
            event_type,
            current_message_count,
            current_unique_channels,
            baseline_avg_messages,
            baseline_std_messages,
            baseline_avg_channels,
            baseline_std_channels,
            baseline_points,
            spike_ratio,
            channel_spike_ratio,
            zscore_messages,
            zscore_channels,
            is_anomaly,
            anomaly_reason,
            computed_at
        FROM telegram_hourly_anomalies
        WHERE hour_bucket = ?
          AND is_anomaly = 1
        ORDER BY spike_ratio DESC, zscore_messages DESC, current_message_count DESC
        LIMIT ?
        """,
        params=[latest_hour, anomaly_limit],
    )


def get_recent_clusters(cluster_window: str, cluster_level: str, cluster_limit: int) -> pd.DataFrame:
    if cluster_limit <= 0:
        return pd.DataFrame()

    return safe_query(
        TELEGRAM_DB,
        """
        SELECT
            event_cluster_id,
            event_type,
            geo_group,
            country,
            location_key,
            location_text,
            cluster_start,
            cluster_end,
            message_count,
            channels,
            channel_count,
            sample_text,
            entities,
            duration_seconds,
            severity_score,
            window_label,
            cluster_level,
            computed_at
        FROM telegram_cluster_metrics_recent
        WHERE window_label = ?
          AND cluster_level = ?
        ORDER BY severity_score DESC, cluster_start DESC
        LIMIT ?
        """,
        params=[cluster_window, cluster_level, cluster_limit],
    )


def get_stock_alerts(stock_limit: int) -> pd.DataFrame:
    if stock_limit <= 0:
        return pd.DataFrame()

    return safe_query(
        MARKETS_DB,
        """
        SELECT *
        FROM stock_alerts
        ORDER BY date DESC
        LIMIT ?
        """,
        params=[stock_limit],
    )


def get_earthquakes(earthquake_limit: int) -> pd.DataFrame:
    if earthquake_limit <= 0:
        return pd.DataFrame()

    return safe_query(
        EARTHQUAKE_DB,
        """
        SELECT
            usgs_id,
            title,
            location,
            type,
            time,
            updated_time,
            magnitude,
            depth_km,
            latitude,
            longitude
        FROM usgs_earthquakes
        ORDER BY time DESC
        LIMIT ?
        """,
        params=[earthquake_limit],
    )


@router.get("/latest")
def get_latest_intel(
    cluster_window: str = Query("last_hour", pattern="^(last_hour|last_day|last_week)$"),
    cluster_level: str = Query("location", pattern="^(location|country)$"),
    anomaly_limit: int = Query(25, ge=0, le=200),
    cluster_limit: int = Query(25, ge=0, le=200),
    stock_limit: int = Query(25, ge=0, le=200),
    earthquake_limit: int = Query(25, ge=0, le=200),
):
    try:
        generated_at = datetime.now(timezone.utc).isoformat()

        latest_hour = get_latest_hour()
        hourly_anomalies = get_hourly_anomalies(latest_hour, anomaly_limit)
        recent_clusters = get_recent_clusters(cluster_window, cluster_level, cluster_limit)
        stock_alerts = get_stock_alerts(stock_limit)
        earthquakes = get_earthquakes(earthquake_limit)

        payload = {
            "generated_at": generated_at,
            "latest_hour": latest_hour,
            "filters": {
                "cluster_window": cluster_window,
                "cluster_level": cluster_level,
                "anomaly_limit": anomaly_limit,
                "cluster_limit": cluster_limit,
                "stock_limit": stock_limit,
                "earthquake_limit": earthquake_limit,
            },
            "summary_counts": {
                "hourly_anomalies": len(hourly_anomalies),
                "recent_clusters": len(recent_clusters),
                "stock_alerts": len(stock_alerts),
                "earthquakes": len(earthquakes),
            },
            "hourly_anomalies": clean_records(hourly_anomalies),
            "recent_clusters": clean_records(recent_clusters),
            "stock_alerts": clean_records(stock_alerts),
            "earthquakes": clean_records(earthquakes),
        }
        return payload

    except Exception:
        logger.exception("get_latest_intel failed")
        raise