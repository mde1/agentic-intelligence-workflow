from fastapi import APIRouter, Query
import sqlite3
import pandas as pd

router = APIRouter()

DB_PATH = "db/telegram_channels.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@router.get("/hourly")
def get_hourly_anomalies(
    anomalies_only: bool = Query(True, description="Return only flagged anomalies"),
    country: str | None = Query(None, description="Filter by country"),
    event_type: str | None = Query(None, description="Filter by event type"),
    min_spike_ratio: float | None = Query(None, description="Minimum spike ratio"),
    limit: int = Query(100, ge=1, le=1000),
):
    conn = get_connection()
    try:
        query = """
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
            WHERE 1=1
        """

        params = []

        if anomalies_only:
            query += " AND is_anomaly = 1"

        if country:
            query += " AND country = ?"
            params.append(country)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if min_spike_ratio is not None:
            query += " AND spike_ratio >= ?"
            params.append(min_spike_ratio)

        query += """
            ORDER BY
                is_anomaly DESC,
                spike_ratio DESC,
                zscore_messages DESC,
                current_message_count DESC
            LIMIT ?
        """
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.get("/hourly/latest-summary")
def get_hourly_anomaly_summary():
    conn = get_connection()
    try:
        latest_hour_query = """
            SELECT MAX(hour_bucket) AS latest_hour
            FROM telegram_hourly_anomalies
        """
        latest_hour_df = pd.read_sql_query(latest_hour_query, conn)
        latest_hour = latest_hour_df.iloc[0]["latest_hour"]

        if latest_hour is None:
            return {
                "latest_hour": None,
                "total_rows": 0,
                "total_anomalies": 0,
                "top_countries": [],
                "top_event_types": [],
            }

        total_query = """
            SELECT COUNT(*) AS total_rows,
                   SUM(is_anomaly) AS total_anomalies
            FROM telegram_hourly_anomalies
            WHERE hour_bucket = ?
        """
        totals = pd.read_sql_query(total_query, conn, params=[latest_hour]).iloc[0]

        countries_query = """
            SELECT
                country,
                COUNT(*) AS anomaly_rows,
                MAX(spike_ratio) AS max_spike_ratio
            FROM telegram_hourly_anomalies
            WHERE hour_bucket = ?
              AND is_anomaly = 1
            GROUP BY country
            ORDER BY anomaly_rows DESC, max_spike_ratio DESC
            LIMIT 10
        """
        countries_df = pd.read_sql_query(countries_query, conn, params=[latest_hour])

        event_types_query = """
            SELECT
                event_type,
                COUNT(*) AS anomaly_rows,
                MAX(spike_ratio) AS max_spike_ratio
            FROM telegram_hourly_anomalies
            WHERE hour_bucket = ?
              AND is_anomaly = 1
            GROUP BY event_type
            ORDER BY anomaly_rows DESC, max_spike_ratio DESC
            LIMIT 10
        """
        event_types_df = pd.read_sql_query(event_types_query, conn, params=[latest_hour])

        return {
            "latest_hour": latest_hour,
            "total_rows": int(totals["total_rows"] or 0),
            "total_anomalies": int(totals["total_anomalies"] or 0),
            "top_countries": countries_df.to_dict(orient="records"),
            "top_event_types": event_types_df.to_dict(orient="records"),
        }
    finally:
        conn.close()