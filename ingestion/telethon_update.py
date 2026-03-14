import os
import sqlite3
import asyncio
from datetime import datetime, timezone
import pandas as pd
from telethon import TelegramClient
from dotenv import load_dotenv
import spacy
import json
import re

load_dotenv()
nlp = spacy.load("en_core_web_sm")

API_ID = int(os.getenv("TELEGRAM_APP_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")

SESSION_NAME = "telegram_session"
DB_PATH = "db/telegram_channels.db"

CHANNELS = [
    "TPOnow",
    "me_observer_TG",
    "rnintel",
    "DDGeopolitics",
    "RezistanceTrench1",
    "tabzlive",
    "presstv",
    "wfwitness",
    "idfofficial",
    "FrontlineReportNews",
    "ClashReport",
    "CIG_telegram",
    "wc_israel",
    "beholdisraelchannel",
    "LebUpdate",
    "FotrosResistancee",
    "VigilantFox",
    "TuckerCarlsonNetwork",
    "thecradlemedia",
    "SGTnewsNetwork",
    "TheIslanderNews",
    "GeoPWatch",
    "OSINTWarfare",
    "aljazeera_world",
    "BellumActaNews",
    "medmannews",
    "gaza911o",
    "ThePakistanNews",
    "DocumentingIsrael",
    "IntelRepublic",
    "hnaftali",
    "intelslava",
    "NewsFromSudan",
    "AssyriaNewsNetwork",
    "kalibrated",
    "sinembargomx",
    "wartimedia",
    "warmonitors",
    "ClaudiaSheinbaum",
    "irna_en",
    "SimurghRes",
    "AMK_MAPPING",
    "tg301military",
    "News_of_Donbass",
    "Wor_mil_news",
    "AlHaqNews",
    "newrulesgeo", 
    "NewsfromSudan",
    "SANANewsEnglish",
    "GeoIndian", 
    "internationalreporters_org",
    "QudsNen",
    "DeepStateUA",
    "DefenderDome",
    "KurdishFrontNews",
    "war_noir",
    "englishabuali",
    "TheTimesOfIsrael2022",
    "The_Jerusalem_Post",
    "ILTVnews",
    "nytimes",
    "MiddleEastEye_TG",
    "trtworld",
]

# =========================
# SPACY SETUP
# =========================
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

patterns = [
    {"label": "GPE", "pattern": "Israel"},
    {"label": "GPE", "pattern": "Iran"},
    {"label": "GPE", "pattern": "Bahrain"},
    {"label": "GPE", "pattern": "Kuwait"},
    {"label": "GPE", "pattern": "Lebanon"},
    {"label": "GPE", "pattern": "Gaza"},
    {"label": "GPE", "pattern": "Tel Aviv"},
    {"label": "GPE", "pattern": "Jerusalem"},
    {"label": "GPE", "pattern": "Haifa"},
    {"label": "GPE", "pattern": "Tehran"},
    {"label": "ORG", "pattern": "IDF"},
    {"label": "ORG", "pattern": "IRGC"},
    {"label": "ORG", "pattern": "CENTCOM"},
    {"label": "ORG", "pattern": "Hezbollah"},
    {"label": "ORG", "pattern": "Hamas"},
    {"label": "ORG", "pattern": "Houthis"},
    {"label": "ORG", "pattern": "jevvs"},
    {"label": "WEAPON", "pattern": "Shahed"},
    {"label": "WEAPON", "pattern": "Shahed drone"},
    {"label": "WEAPON", "pattern": "ballistic missile"},
    {"label": "WEAPON", "pattern": "cruise missile"},
    {"label": "WEAPON", "pattern": "drone"},
    {"label": "WEAPON", "pattern": "UAV"},
    {"label": "SYSTEM", "pattern": "Iron Dome"},
    {"label": "SYSTEM", "pattern": "Arrow"},
    {"label": "SYSTEM", "pattern": "David's Sling"},
]

# Add patterns only once
existing_patterns = getattr(ruler, "patterns", [])
if not existing_patterns:
    ruler.add_patterns(patterns)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS telegram_messages (
    channel_username TEXT NOT NULL,
    channel_title TEXT,
    message_id INTEGER NOT NULL,
    message_date TEXT,
    message_text TEXT,
    scraped_at TEXT,
    message_hour INTEGER,
    message_month INTEGER,
    message_year INTEGER,
    message_minute INTEGER,
    message_dateFormatted TEXT,
    message_time TEXT,
    tagged_entities TEXT,
    entity_json TEXT,
    is_event INTEGER,
    event_type TEXT,
    PRIMARY KEY (channel_username, message_id)
);
"""

CREATE_INDEX_1 = """
CREATE INDEX IF NOT EXISTS idx_channel_message_id
ON telegram_messages(channel_username, message_id);
"""

CREATE_INDEX_2 = """
CREATE INDEX IF NOT EXISTS idx_channel_message_date
ON telegram_messages(channel_username, message_date);
"""


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(CREATE_INDEX_1)
        conn.execute(CREATE_INDEX_2)
        conn.commit()


def get_last_message_id(db_path: str, channel_username: str) -> int | None:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT MAX(message_id)
            FROM telegram_messages
            WHERE channel_username = ?
            """,
            (channel_username,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def insert_messages(db_path: str, rows: list[dict]) -> None:
    if not rows:
        return

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO telegram_messages (
                channel_username,
                channel_title,
                message_id,
                message_date,
                message_text,
                scraped_at,
                message_hour,
                message_month,
                message_year,
                message_minute,
                message_dateFormatted,
                message_time,
                tagged_entities,
                entity_json,
                is_event,
                event_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
            (
                row.get("channel_username"),
                row.get("channel_title"),
                row.get("message_id"),
                row.get("message_date"),
                row.get("message_text"),
                row.get("scraped_at"),
                row.get("message_hour"),
                row.get("message_month"),
                row.get("message_year"),
                row.get("message_minute"),
                row.get("message_dateFormatted"),
                row.get("message_time"),
                row.get("tagged_entities"),
                row.get("entity_json"),
                row.get("is_event"),
                row.get("event_type")
            )
                for row in rows
            ],
        )
        conn.commit()

# =========================
# ENRICHMENT
# =========================
def extract_entities(text: str) -> tuple[str | None, str | None]:
    if not text or not str(text).strip():
        return None, None

    doc = nlp(str(text))

    entities = []
    seen = set()

    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_label = ent.label_
        key = (ent_text.lower(), ent_label)

        if ent_text and key not in seen:
            seen.add(key)
            entities.append({
                "text": ent_text,
                "label": ent_label
            })

    tagged_entities = ", ".join(
        f'{e["text"]} ({e["label"]})' for e in entities
    ) if entities else None

    entity_json = json.dumps(entities, ensure_ascii=False) if entities else None

    return tagged_entities, entity_json

def normalize_message_dt(value):
    if value is None:
        return None

    # pandas Timestamp
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            return value.tz_localize("UTC").to_pydatetime()
        return value.tz_convert("UTC").to_pydatetime()

    # plain datetime
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    # plain date (rare, but safe to handle)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)

    # string fallback
    if isinstance(value, str):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()

    return None

EVENT_PATTERN = re.compile(
                        r"\b("
                        r"impact|hit|strike|struck|interception|intercepted|"
                        r"sirens?|explosion|explosions|drone|missile|rocket|launch|launched"
                        r")\b",
                        re.IGNORECASE
                    )

def classify_event(text: str) -> str | None:
    if not text or not str(text).strip():
        return None

    text = str(text).lower()

    if "intercept" in text:
        return "interception"
    if "siren" in text:
        return "siren"
    if "impact" in text or "hit" in text or "strike" in text or "struck" in text:
        return "impact"
    if "launch" in text or "launched" in text:
        return "launch"
    if "explosion" in text:
        return "explosion"
    if "drone" in text or "missile" in text or "rocket" in text:
        return "projectile"

    return "other"


def enrich_message_row(
    channel_username: str,
    channel_title: str,
    msg,
    scraped_at: str
        ) -> dict:

    message_text = msg.text
    is_event = int(bool(EVENT_PATTERN.search(message_text or "")))
    event_type = classify_event(message_text) if is_event else None
    message_dt = msg.date#.isoformat() if msg.date else None

    # Normalize timezone just in case
    message_dt = normalize_message_dt(msg.date)

    tagged_entities, entity_json = extract_entities(message_text)

    return {
        "channel_username": channel_username,
        "channel_title": channel_title,
        "message_id": msg.id,
        "message_date": message_dt.isoformat() if message_dt is not None else None,
        "message_text": message_text,
        "scraped_at": scraped_at,
        "message_hour": message_dt.hour if message_dt is not None else None,
        "message_month": message_dt.month if message_dt is not None else None,
        "message_year": message_dt.year if message_dt is not None else None,
        "message_minute": message_dt.minute if message_dt is not None else None,
        "message_dateFormatted": message_dt.date().isoformat() if message_dt is not None else None,
        "message_time": message_dt.time().isoformat() if message_dt is not None else None,
        "tagged_entities": tagged_entities,
        "entity_json": entity_json,
        "is_event": is_event,
        "event_type": event_type,
    }

async def update_channel(client: TelegramClient, db_path: str, channel_username: str, backfill_limit=3000) -> list[dict]:
    entity = await client.get_entity(channel_username)
    channel_title = getattr(entity, "title", channel_username)

    last_message_id = get_last_message_id(db_path, channel_username)
    print(f"{channel_username}: last saved message_id = {last_message_id}")

    rows = []
    now_utc = datetime.now(timezone.utc).isoformat()

    # min_id means: only return messages with ID > min_id
    # CASE 1 — channel already in database
    if last_message_id:
        print(f"{channel_username}: fetching messages after ID {last_message_id}")

        async for msg in client.iter_messages(entity, min_id=last_message_id or 0):
            row = enrich_message_row(
                channel_username=channel_username,
                channel_title=channel_title,
                msg=msg,
                scraped_at=now_utc
            )

            if row["message_date"] is not None:
                rows.append(row)
            else:
                print(f"Skipping message {msg.id}: missing message_date")
    # CASE 2 — first time seeing this channel
    else:
        print(f"{channel_username}: no history found, pulling last {backfill_limit}")

        async for msg in client.iter_messages(entity, limit=backfill_limit):
            row = enrich_message_row(
                channel_username=channel_username,
                channel_title=channel_title,
                msg=msg,
                scraped_at=now_utc
            )

            if row["message_date"] is not None:
                rows.append(row)
            else:
                print(f"Skipping message {msg.id}: missing message_date")

    return rows


async def main():
    init_db(DB_PATH)

    async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
        for channel in CHANNELS:
            try:
                print(f"\nChecking channel: {channel}")
                rows = await update_channel(client, DB_PATH, channel, backfill_limit=3000)
                insert_messages(DB_PATH, rows)
                print(f"Inserted {len(rows)} rows for {channel}")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Failed for {channel}: {e}")


if __name__ == "__main__":
    asyncio.run(main())