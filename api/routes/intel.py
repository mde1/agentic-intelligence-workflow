from fastapi import APIRouter, Query
import sqlite3
import pandas as pd
from datetime import datetime, timezone

router = APIRouter()

DB_PATH = "db/telegram_channels.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def safe_query(conn: sqlite3.Connection, query: str, params=None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    except Exception:
        return pd.DataFrame()


@router.get("/latest")
def get_latest_intel(
    cluster_window: str = Query("last_hour", pattern="^(last_hour|last_day)$"),
    cluster_level: str = Query("location", pattern="^(location|country)$"),
    anomaly_limit: int = Query(25, ge=0, le=200),
    cluster_limit: int = Query(25, ge=0, le=200),
    stock_limit: int = Query(25, ge=0, le=200),
    earthquake_limit: int = Query(25, ge=0, le=200),
):
    conn = get_connection()
    try:
        generated_at = datetime.now(timezone.utc).isoformat()

        # ------------------------------------------------------------
        # Latest hour from hourly anomalies
        # ------------------------------------------------------------
        latest_hour_df = safe_query(
            conn,
            """
            SELECT MAX(hour_bucket) AS latest_hour
            FROM telegram_hourly_anomalies
            """
        )
        latest_hour = None
        if not latest_hour_df.empty:
            latest_hour = latest_hour_df.iloc[0]["latest_hour"]

        # ------------------------------------------------------------
        # Hourly anomalies
        # ------------------------------------------------------------
        hourly_anomalies = pd.DataFrame()
        if latest_hour is not None:
            hourly_anomalies = safe_query(
                conn,
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
                params=[latest_hour, anomaly_limit]
            )

        # ------------------------------------------------------------
        # Recent clusters
        # ------------------------------------------------------------
        recent_clusters = safe_query(
            conn,
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
            params=[cluster_window, cluster_level, cluster_limit]
        )

        # ------------------------------------------------------------
        # Stock alerts
        # Assumes you created a stock_alerts table.
        # If not present yet, this safely returns [].
        # ------------------------------------------------------------
        stock_alerts = safe_query(
            conn,
            """
            SELECT *
            FROM stock_alerts
            ORDER BY date DESC
            LIMIT ?
            """,
            params=[stock_limit]
        )

        # ------------------------------------------------------------
        # Earthquakes
        # Assumes an earthquakes table.
        # If not present yet, this safely returns [].
        # ------------------------------------------------------------
        earthquakes = safe_query(
            conn,
            """
            SELECT *
            FROM earthquakes
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params=[earthquake_limit]
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
    finally:
        conn.close()