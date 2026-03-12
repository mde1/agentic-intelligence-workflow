from fastapi import APIRouter
import sqlite3
import pandas as pd

router = APIRouter()

DB_PATH = "db/telegram_channels.db"

@router.get("/clusters/{window_label}")
def get_clusters(window_label: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
            SELECT *
            FROM telegram_cluster_metrics_recent
            WHERE window_label = ?
            ORDER BY severity_score DESC, cluster_start DESC
        """
        df = pd.read_sql_query(query, conn, params=(window_label,))
        return df.to_dict(orient="records")
    finally:
        conn.close()