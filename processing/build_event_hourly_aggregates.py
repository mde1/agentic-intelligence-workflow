import sqlite3
import pandas as pd
from datetime import datetime, timezone
import re

DB_PATH = "db/telegram_channels.db"
OUTPUT_TABLE = "telegram_event_hourly"

LOCATION_ALIASES = {
    "kyiv": {"country": "Ukraine", "location_key": "kyiv"},
    "kiev": {"country": "Ukraine", "location_key": "kyiv"},
    "kharkiv": {"country": "Ukraine", "location_key": "kharkiv"},
    "donetsk": {"country": "Ukraine", "location_key": "donetsk"},
    "crimea": {"country": "Ukraine", "location_key": "crimea"},
    "ukraine": {"country": "Ukraine", "location_key": None},

    "erbil": {"country": "Iraq", "location_key": "erbil"},
    "baghdad": {"country": "Iraq", "location_key": "baghdad"},
    "basra": {"country": "Iraq", "location_key": "basra"},
    "iraq": {"country": "Iraq", "location_key": None},

    "tehran": {"country": "Iran", "location_key": "tehran"},
    "iran": {"country": "Iran", "location_key": None},

    "tel aviv": {"country": "Israel", "location_key": "tel_aviv"},
    "jerusalem": {"country": "Israel", "location_key": "jerusalem"},
    "haifa": {"country": "Israel", "location_key": "haifa"},
    "eilat": {"country": "Israel", "location_key": "eilat"},
    "israel": {"country": "Israel", "location_key": None},

    "gaza city": {"country": "Palestine", "location_key": "gaza"},
    "gaza": {"country": "Palestine", "location_key": "gaza"},
    "west bank": {"country": "Palestine", "location_key": "west_bank"},
    "palestine": {"country": "Palestine", "location_key": None},

    "beirut": {"country": "Lebanon", "location_key": "beirut"},
    "tyre": {"country": "Lebanon", "location_key": "tyre"},
    "lebanon": {"country": "Lebanon", "location_key": None},

    "damascus": {"country": "Syria", "location_key": "damascus"},
    "aleppo": {"country": "Syria", "location_key": "aleppo"},
    "syria": {"country": "Syria", "location_key": None},

    "dubai": {"country": "UAE", "location_key": "dubai"},
    "uae": {"country": "UAE", "location_key": None},

    "bahrain": {"country": "Bahrain", "location_key": None},
}

SORTED_LOCATION_TERMS = sorted(LOCATION_ALIASES.keys(), key=len, reverse=True)


def extract_geography(text: str) -> dict:
    if not text or not str(text).strip():
        return {
            "country": None,
            "location_key": None,
            "location_text": None,
        }

    text_lower = str(text).lower()

    for term in SORTED_LOCATION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            info = LOCATION_ALIASES[term]
            return {
                "country": info["country"],
                "location_key": info["location_key"],
                "location_text": term,
            }

    return {
        "country": None,
        "location_key": None,
        "location_text": None,
    }


def load_events(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            message_id,
            channel_username,
            message_date,
            message_text,
            event_type,
            is_event
        FROM telegram_messages
        WHERE is_event = 1
          AND message_date IS NOT NULL
          AND event_type IS NOT NULL
    """
    return pd.read_sql_query(query, conn)


def build_hourly_aggregates(events: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "hour_bucket",
        "country",
        "event_type",
        "message_count",
        "unique_channels",
        "computed_at",
    ]

    if events.empty:
        return pd.DataFrame(columns=output_columns)

    events = events.copy()
    events["message_date"] = pd.to_datetime(events["message_date"], format="ISO8601", utc=True, errors="coerce")
    events = events.dropna(subset=["message_date", "event_type", "message_text"])

    geo_df = events["message_text"].apply(extract_geography).apply(pd.Series)
    events = pd.concat([events, geo_df], axis=1)

    events["country"] = events["country"].fillna("Unknown")
    events["hour_bucket"] = events["message_date"].dt.floor("h")

    hourly = (
        events.groupby(["hour_bucket", "country", "event_type"], as_index=False)
        .agg(
            message_count=("message_id", "count"),
            unique_channels=("channel_username", "nunique"),
        )
    )

    hourly["computed_at"] = datetime.now(timezone.utc).isoformat()

    return hourly[output_columns]


def ensure_output_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    cursor.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} (
            hour_bucket TEXT,
            country TEXT,
            event_type TEXT,
            message_count INTEGER,
            unique_channels INTEGER,
            computed_at TEXT
        )
    """)
    conn.commit()


def save_hourly_aggregates(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if not df.empty:
        df.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_hour
        ON {OUTPUT_TABLE}(hour_bucket)
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_country_event
        ON {OUTPUT_TABLE}(country, event_type)
    """)
    conn.commit()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        events = load_events(conn)
        hourly = build_hourly_aggregates(events)
        ensure_output_table(conn)
        save_hourly_aggregates(conn, hourly)
        print(f"Wrote {len(hourly)} hourly aggregate rows to {OUTPUT_TABLE}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()