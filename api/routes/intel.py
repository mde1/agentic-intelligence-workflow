from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"

TELEGRAM_DB_PATH = DB_DIR / "telegram_channels.db"
MARKET_DB_PATH = DB_DIR / "market_data.db"
MARKET_OSINT_DB_PATH = DB_DIR / "market_osint.db"

# Tables grouped by source so we only open the DBs that are actually needed.
TELEGRAM_REQUIRED_TABLES = (
    "telegram_hourly_anomalies",
    "telegram_cluster_metrics_recent",
)
MARKET_REQUIRED_TABLES = (
    "stock_alerts",
    "stocks_data_long",
)
MARKET_OSINT_REQUIRED_TABLES = (
    "usgs_earthquakes",
)

# Whitelisted ORDER BY fields so we never interpolate arbitrary user input into SQL.
_ORDERABLE_TABLE_COLUMNS: dict[str, set[str]] = {
    "telegram_hourly_anomalies": {
        "hour_bucket",
        "spike_ratio",
        "zscore_messages",
        "current_message_count",
        "computed_at",
    },
    "telegram_cluster_metrics_recent": {
        "severity_score",
        "cluster_start",
        "cluster_end",
        "message_count",
        "computed_at",
    },
    "stock_alerts": {"Date", "Ticker", "Alert"},
    "stocks_data_long": {"Date", "Ticker", "Price"},
    "usgs_earthquakes": {"time", "updated_time", "magnitude", "depth_km"},
}

def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def db_exists(db_path: Path) -> bool:
    return db_path.exists() and db_path.is_file()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1"
    row = conn.execute(query, (table_name,)).fetchone()
    return row is not None


def safe_query(conn: sqlite3.Connection, query: str, params: list[Any] | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    except Exception:
        logger.exception("SQL query failed")
        return pd.DataFrame()

def safe_table_query(
    conn: sqlite3.Connection | None,
    table_name: str,
    *,
    columns: list[str] | None = None,
    where_clause: str | None = None,
    params: list[Any] | None = None,
    order_by: str | None = None,
    descending: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    if conn is None:
        return pd.DataFrame()

    if not table_exists(conn, table_name):
        logger.warning("Table %s not found in connected database", table_name)
        return pd.DataFrame()

    select_columns = ", ".join(columns) if columns else "*"
    query = [f"SELECT {select_columns} FROM {table_name}"]
    sql_params = list(params or [])

    if where_clause:
        query.append(f"WHERE {where_clause}")

    if order_by:
        allowed_columns = _ORDERABLE_TABLE_COLUMNS.get(table_name, set())
        if order_by not in allowed_columns:
            raise ValueError(f"Unsupported order_by '{order_by}' for table '{table_name}'")
        direction = "DESC" if descending else "ASC"
        query.append(f"ORDER BY {order_by} {direction}")

    if limit is not None:
        query.append("LIMIT ?")
        sql_params.append(limit)

    return safe_query(conn, "\n".join(query), sql_params)


def open_optional_connection(db_path: Path, required_tables: tuple[str, ...]) -> sqlite3.Connection | None:
    if not db_exists(db_path):
        logger.warning("Database file not found: %s", db_path)
        return None

    conn = get_connection(db_path)
    existing_tables = {t for t in required_tables if table_exists(conn, t)}
    if not existing_tables:
        logger.warning(
            "Database %s opened but none of the required tables were found: %s",
            db_path,
            ", ".join(required_tables),
        )
    return conn

@router.get("/latest")
def get_latest_intel(
    cluster_window: str = Query("last_hour", pattern="^(last_hour|last_day)$"),
    cluster_level: str = Query("location", pattern="^(location|country)$"),
    anomaly_limit: int = Query(25, ge=1, le=200),
    cluster_limit: int = Query(25, ge=1, le=200),
    stock_limit: int = Query(25, ge=1, le=200),
    stock_price_limit: int = Query(100, ge=1, le=1000),
    earthquake_limit: int = Query(25, ge=1, le=200),
):
    telegram_conn: sqlite3.Connection | None = None
    market_conn: sqlite3.Connection | None = None
    market_osint_conn: sqlite3.Connection | None = None

    try:
        telegram_conn = open_optional_connection(TELEGRAM_DB_PATH, TELEGRAM_REQUIRED_TABLES)
        market_conn = open_optional_connection(MARKET_DB_PATH, MARKET_REQUIRED_TABLES)
        market_osint_conn = open_optional_connection(MARKET_OSINT_DB_PATH, MARKET_OSINT_REQUIRED_TABLES)

        generated_at = datetime.now(timezone.utc).isoformat()

        latest_hour = None
        latest_hour_df = safe_table_query(
            telegram_conn,
            "telegram_hourly_anomalies",
            columns=["MAX(hour_bucket) AS latest_hour"],
        )
        if not latest_hour_df.empty:
            latest_hour = latest_hour_df.iloc[0]["latest_hour"]

        hourly_anomalies = pd.DataFrame()
        if latest_hour is not None:
            hourly_anomalies = safe_table_query(
                telegram_conn,
                "telegram_hourly_anomalies",
                columns=[
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
                ],
                where_clause="hour_bucket = ? AND is_anomaly = 1",
                params=[latest_hour],
                order_by="spike_ratio",
                descending=True,
                limit=anomaly_limit,
            )

        recent_clusters = safe_table_query(
            telegram_conn,
            "telegram_cluster_metrics_recent",
            columns=[
                "event_cluster_id",
                "event_type",
                "geo_group",
                "country",
                "location_key",
                "location_text",
                "cluster_start",
                "cluster_end",
                "message_count",
                "channels",
                "channel_count",
                "sample_text",
                "entities",
                "duration_seconds",
                "severity_score",
                "window_label",
                "cluster_level",
                "computed_at",
            ],
            where_clause="window_label = ? AND cluster_level = ?",
            params=[cluster_window, cluster_level],
            order_by="severity_score",
            descending=True,
            limit=cluster_limit,
        )

        stock_alerts = safe_table_query(
            market_conn,
            "stock_alerts",
            order_by="Date",
            descending=True,
            limit=stock_limit,
        )

        # Included for downstream charting/trend analysis. This does not replace stock_alerts.
        stock_prices = safe_table_query(
            market_conn,
            "stocks_data_long",
            order_by="Date",
            descending=True,
            limit=stock_price_limit,
        )

        earthquakes = safe_table_query(
            market_osint_conn,
            "usgs_earthquakes",
            order_by="time",
            descending=True,
            limit=earthquake_limit,
        )

        available_sources = {
            "telegram": telegram_conn is not None,
            "market_data": market_conn is not None,
            "market_osint": market_osint_conn is not None,
        }

        available_tables = {
            "telegram_hourly_anomalies": bool(telegram_conn and table_exists(telegram_conn, "telegram_hourly_anomalies")),
            "telegram_cluster_metrics_recent": bool(telegram_conn and table_exists(telegram_conn, "telegram_cluster_metrics_recent")),
            "stock_alerts": bool(market_conn and table_exists(market_conn, "stock_alerts")),
            "stocks_data_long": bool(market_conn and table_exists(market_conn, "stocks_data_long")),
            "usgs_earthquakes": bool(market_osint_conn and table_exists(market_osint_conn, "usgs_earthquakes")),
        }

        return {
            "generated_at": generated_at,
            "latest_hour": latest_hour,
            "filters": {
                "cluster_window": cluster_window,
                "cluster_level": cluster_level,
                "anomaly_limit": anomaly_limit,
                "cluster_limit": cluster_limit,
                "stock_limit": stock_limit,
                "stock_price_limit": stock_price_limit,
                "earthquake_limit": earthquake_limit,
            },
            "available_sources": available_sources,
            "available_tables": available_tables,
            "summary_counts": {
                "hourly_anomalies": len(hourly_anomalies),
                "recent_clusters": len(recent_clusters),
                "stock_alerts": len(stock_alerts),
                "stock_prices": len(stock_prices),
                "earthquakes": len(earthquakes),
            },
            "hourly_anomalies": hourly_anomalies.to_dict(orient="records"),
            "recent_clusters": recent_clusters.to_dict(orient="records"),
            "stock_alerts": stock_alerts.to_dict(orient="records"),
            "stock_prices": stock_prices.to_dict(orient="records"),
            "earthquakes": earthquakes.to_dict(orient="records"),
        }
    finally:
        for conn in (telegram_conn, market_conn, market_osint_conn):
            if conn is not None:
                conn.close()