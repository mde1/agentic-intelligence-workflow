import sqlite3
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

FEED_URL = "https://www.twz.com/feed"
DB_PATH = "db/twz_feed.db"


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT,
            title TEXT,
            link TEXT NOT NULL,
            published TEXT,
            summary TEXT,
            author TEXT,
            category TEXT,
            fetched_at TEXT NOT NULL,
            raw_entry TEXT,
            UNIQUE(link)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feed_state (
            feed_url TEXT PRIMARY KEY,
            last_checked TEXT,
            last_successful_fetch TEXT
        )
    """)
    conn.commit()


def parse_published(entry) -> str | None:
    # Best effort to normalize published date into ISO format
    for key in ("published", "updated"):
        value = entry.get(key)
        if value:
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                return value
    return None


def extract_category(entry) -> str | None:
    tags = entry.get("tags", [])
    if not tags:
        return None
    terms = [t.get("term") for t in tags if t.get("term")]
    return ", ".join(terms) if terms else None


def upsert_entries(conn: sqlite3.Connection, feed) -> tuple[int, int]:
    inserted = 0
    updated = 0
    fetched_at = datetime.now(timezone.utc).isoformat()

    for entry in feed.entries:
        guid = entry.get("id") or entry.get("guid")
        title = entry.get("title")
        link = entry.get("link")
        published = parse_published(entry)
        summary = entry.get("summary") or entry.get("description")
        author = entry.get("author")
        category = extract_category(entry)
        raw_entry = str(dict(entry))

        if not link:
            continue

        # Try insert first
        cur = conn.execute("""
            INSERT OR IGNORE INTO posts
            (guid, title, link, published, summary, author, category, fetched_at, raw_entry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (guid, title, link, published, summary, author, category, fetched_at, raw_entry))

        if cur.rowcount == 1:
            inserted += 1
        else:
            # Optional: update existing row if feed data changed
            cur = conn.execute("""
                UPDATE posts
                SET guid = COALESCE(?, guid),
                    title = COALESCE(?, title),
                    published = COALESCE(?, published),
                    summary = COALESCE(?, summary),
                    author = COALESCE(?, author),
                    category = COALESCE(?, category),
                    fetched_at = ?,
                    raw_entry = ?
                WHERE link = ?
            """, (guid, title, published, summary, author, category, fetched_at, raw_entry, link))
            if cur.rowcount == 1:
                updated += 1

    conn.commit()
    return inserted, updated


def update_feed_state(conn: sqlite3.Connection, feed_url: str, success: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()

    if success:
        conn.execute("""
            INSERT INTO feed_state (feed_url, last_checked, last_successful_fetch)
            VALUES (?, ?, ?)
            ON CONFLICT(feed_url) DO UPDATE SET
                last_checked = excluded.last_checked,
                last_successful_fetch = excluded.last_successful_fetch
        """, (feed_url, now, now))
    else:
        conn.execute("""
            INSERT INTO feed_state (feed_url, last_checked, last_successful_fetch)
            VALUES (?, ?, NULL)
            ON CONFLICT(feed_url) DO UPDATE SET
                last_checked = excluded.last_checked
        """, (feed_url, now))
    conn.commit()


def fetch_and_store(feed_url: str = FEED_URL, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)

        feed = feedparser.parse(feed_url)

        # feed.bozo == 1 means parser noticed a problem, but many feeds still parse fine
        if getattr(feed, "status", None) and feed.status >= 400:
            update_feed_state(conn, feed_url, success=False)
            raise RuntimeError(f"Feed request failed with HTTP status {feed.status}")

        inserted, updated = upsert_entries(conn, feed)
        update_feed_state(conn, feed_url, success=True)

        print(f"Feed title: {feed.feed.get('title', 'Unknown')}")
        print(f"Entries fetched: {len(feed.entries)}")
        print(f"Inserted new rows: {inserted}")
        print(f"Updated existing rows: {updated}")
        print(f"SQLite DB: {db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    fetch_and_store()