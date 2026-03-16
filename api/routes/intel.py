from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEGRAM_DB = PROJECT_ROOT / "db" / "telegram_channels.db"
MARKET_DB = PROJECT_ROOT / "db" / "market_data.db"
MARKET_OSINT_DB = PROJECT_ROOT / "db" / "market_osint.db"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def safe_query(conn: sqlite3.Connection, query: str, params: list | tuple | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    except Exception:
        return pd.DataFrame()


@router.get("/latest")
def get_latest_intel(
    cluster_window: str = Query("last_hour", pattern="^(last_hour|last_day|last_week)$"),
    cluster_level: str = Query("location", pattern="^(location|country)$"),
    anomaly_limit: int = Query(25, ge=1, le=200),
    cluster_limit: int = Query(25, ge=1, le=200),
    stock_limit: int = Query(25, ge=1, le=200),
    earthquake_limit: int = Query(25, ge=1, le=200),
):
    generated_at = datetime.now(timezone.utc).isoformat()

    with get_connection(TELEGRAM_DB) as telegram_conn, get_connection(MARKET_DB) as market_conn, get_connection(MARKET_OSINT_DB) as osint_conn:
        latest_hour_df = safe_query(
            telegram_conn,
            """
            SELECT MAX(hour_bucket) AS latest_hour
            FROM telegram_hourly_anomalies
            """,
        )
        latest_hour = None
        if not latest_hour_df.empty:
            latest_hour = latest_hour_df.iloc[0]["latest_hour"]

        hourly_anomalies = pd.DataFrame()
        if latest_hour is not None:
            hourly_anomalies = safe_query(
                telegram_conn,
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
                ORDER BY
                    spike_ratio DESC,
                    zscore_messages DESC,
                    current_message_count DESC
                LIMIT ?
                """,
                params=[latest_hour, anomaly_limit],
            )

        recent_clusters = safe_query(
            telegram_conn,
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

        stock_alerts = safe_query(
            market_conn,
            """
            SELECT *
            FROM stock_alerts
            ORDER BY Date DESC
            LIMIT ?
            """,
            params=[stock_limit],
        )

        earthquakes = safe_query(
            osint_conn,
            """
            SELECT *
            FROM usgs_earthquakes
            ORDER BY time DESC
            LIMIT ?
            """,
            params=[earthquake_limit],
        )

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
        "hourly_anomalies": hourly_anomalies.to_dict(orient="records"),
        "recent_clusters": recent_clusters.to_dict(orient="records"),
        "stock_alerts": stock_alerts.to_dict(orient="records"),
        "earthquakes": earthquakes.to_dict(orient="records"),
    }
    return payload
