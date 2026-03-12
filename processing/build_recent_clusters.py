import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
import re

DB_PATH = "db/telegram_channels.db"
OUTPUT_TABLE = "telegram_cluster_metrics_recent"


LOCATION_ALIASES = {
    "kyiv": {"country": "Ukraine", "location_key": "kyiv"},
    "kiev": {"country": "Ukraine", "location_key": "kyiv"},
    "kharkiv": {"country": "Ukraine", "location_key": "kharkiv"},
    "donetsk": {"country": "Ukraine", "location_key": "donetsk"},
    "crimea": {"country": "Ukraine", "location_key": "crimea"},

    "erbil": {"country": "Iraq", "location_key": "erbil"},
    "baghdad": {"country": "Iraq", "location_key": "baghdad"},
    "basra": {"country": "Iraq", "location_key": "basra"},
    "iraq": {"country": "Iraq", "location_key": None},

    "tehran": {"country": "Iran", "location_key":'tehran'},
    "hormuz": {"country": "Iran", "location_key":"hormuz"},
    "kashan": {"country": "Iran", "location_key": "kashan"},

    "tel aviv": {"country": "Israel", "location_key": "tel_aviv"},
    "jerusalem": {"country": "Israel", "location_key": "jerusalem"},
    "haifa": {"country": "Israel", "location_key": "haifa"},
    "israel": {"country": "Israel", "location_key": None},
    "eilat": {"country": "Israel", "location_key": "eilat"},
    "settlement":{"country": "Israel", "location_key": "settlement"},
    "galilee": {"country":"Israel", "location_key": "galilee"},

    "gaza": {"country": "Palestine", "location_key": "gaza"},
    "gaza city": {"country": "Palestine", "location_key": "gaza"},
    "west bank": {"country": "Palestine", "location_key": "west_bank"},

    "beirut": {"country": "Lebanon", "location_key": "beirut"},
    "tyre": {"country": "Lebanon", "location_key": "tyre"},
    "lebanon": {"country": "Lebanon", "location_key": None},
    "ansar": {"country":"Lebanon", "location_key":"ansar"},

    "damascus": {"country": "Syria", "location_key": "damascus"},
    "aleppo": {"country": "Syria", "location_key": "aleppo"},
    "syria": {"country": "Syria", "location_key": None},

    "dubai": {"country": "UAE", "location_key": "dubai"},
    "uae": {"country": "UAE", "location_key":"uae"},
    "al dhafra": {"country": "UAE", "location_key":"al dhafra"},
    
    "bahrain": {"country": "bahrain", "location_key": None},


}

SORTED_LOCATION_TERMS = sorted(LOCATION_ALIASES.keys(), key=len, reverse=True)


def extract_geography(text: str) -> dict:
    if not text or not str(text).strip():
        return {
            "country": None,
            "location_key": None,
            "location_text": None
        }

    text_lower = str(text).lower()

    matches = []
    for term in SORTED_LOCATION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            matches.append(term)

    if not matches:
        return {
            "country": None,
            "location_key": None,
            "location_text": None
        }

    # Prefer the first longest/specific match
    best = matches[0]
    info = LOCATION_ALIASES[best]

    return {
        "country": info["country"],
        "location_key": info["location_key"],
        "location_text": best
    }



def load_recent_events(conn: sqlite3.Connection, hours: int) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    query = """
        SELECT
            message_id,
            channel_username,
            channel_title,
            message_date,
            message_text,
            tagged_entities,
            event_type,
            is_event
        FROM telegram_messages
        WHERE is_event = 1
          AND message_date IS NOT NULL
          AND message_date >= ?
    """

    df = pd.read_sql_query(query, conn, params=(cutoff,))
    return df


def build_clusters(events: pd.DataFrame, window_label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "event_cluster_id", "event_type", "cluster_start", "cluster_end",
            "message_count", "channels", "channel_count", "sample_text",
            "entities", "duration_seconds", "severity_score",
            "window_label", "computed_at"
        ])

    events = events.copy()
    events["message_date"] = pd.to_datetime(events["message_date"], utc=True, errors="coerce")
    events = events.dropna(subset=["message_date", "event_type"])
    events = events.sort_values(["event_type", "message_date"])

    events["time_diff_seconds"] = (
        events.groupby("event_type")["message_date"]
        .diff()
        .dt.total_seconds()
    )

    events["new_cluster"] = (
        events["time_diff_seconds"].isna() |
        (events["time_diff_seconds"] > 30)
    )

    events["cluster_num"] = (
        events.groupby("event_type")["new_cluster"]
        .cumsum()
    )

    events["event_cluster_id"] = (
        events["event_type"].astype(str) + "_" + events["cluster_num"].astype(str)
    )

    cluster_summary = (
        events.groupby(["event_cluster_id", "event_type"], as_index=False)
        .agg(
            cluster_start=("message_date", "min"),
            cluster_end=("message_date", "max"),
            message_count=("message_id", "count"),
            channels=("channel_username", lambda x: ", ".join(sorted(set(v for v in x.dropna() if v)))),
            channel_count=("channel_username", "nunique"),
            sample_text=("message_text", "first"),
            entities=("tagged_entities", lambda x: ", ".join(sorted(set(v for v in x.dropna() if v)))),
        )
    )

    cluster_summary["duration_seconds"] = (
        cluster_summary["cluster_end"] - cluster_summary["cluster_start"]
    ).dt.total_seconds()

    cluster_summary["severity_score"] = (
        cluster_summary["message_count"] + 2 * cluster_summary["channel_count"]
    )

    cluster_summary["window_label"] = window_label
    cluster_summary["computed_at"] = datetime.now(timezone.utc).isoformat()

    cluster_summary = cluster_summary.sort_values(
        ["severity_score", "cluster_start"],
        ascending=[False, False]
    )

    return cluster_summary


def save_clusters(conn: sqlite3.Connection, df: pd.DataFrame, window_label: str) -> None:
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {OUTPUT_TABLE} WHERE window_label = ?", (window_label,))
    conn.commit()

    df.to_sql(OUTPUT_TABLE, conn, if_exists="append", index=False)


def ensure_output_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            event_cluster_id TEXT,
            event_type TEXT,
            cluster_start TEXT,
            cluster_end TEXT,
            message_count INTEGER,
            channels TEXT,
            channel_count INTEGER,
            sample_text TEXT,
            entities TEXT,
            duration_seconds REAL,
            severity_score REAL,
            window_label TEXT,
            computed_at TEXT
        )
    """)
    conn.commit()


def run_for_window(conn: sqlite3.Connection, hours: int, window_label: str) -> None:
    events = load_recent_events(conn, hours=hours)
    clusters = build_clusters(events, window_label=window_label)
    save_clusters(conn, clusters, window_label=window_label)
    print(f"{window_label}: wrote {len(clusters)} clusters")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_output_table(conn)
        run_for_window(conn, hours=1, window_label="last_hour")
        run_for_window(conn, hours=24, window_label="last_day")
    finally:
        conn.close()


if __name__ == "__main__":
    main()