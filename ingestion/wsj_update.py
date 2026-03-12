import sqlite3
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

DB_PATH = "db/news_feeds.db"

FEED_URLS = [
    "https://feeds.content.dowjones.io/public/rss/socialeconomyfeed",
    "https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
    "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "https://www.aljazeera.com/xml/rss/all.xml",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_url TEXT NOT NULL,
            feed_title TEXT,
            guid TEXT,
            title TEXT,
            link TEXT NOT NULL,
            published TEXT,
            summary TEXT,
            author TEXT,
            category TEXT,
            fetched_at TEXT NOT NULL,
            raw_entry TEXT,
            UNIQUE(feed_url, link)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feed_state (
            feed_url TEXT PRIMARY KEY,
            feed_title TEXT,
            last_checked TEXT,
            last_successful_fetch TEXT,
            last_status INTEGER,
            last_error TEXT
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_published ON posts(published)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_feed_url ON posts(feed_url)")
    conn.commit()


def parse_datetime(entry) -> str | None:
    for key in ("published", "updated", "created"):
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
    terms = [tag.get("term") for tag in tags if tag.get("term")]
    return ", ".join(terms) if terms else None


def update_feed_state(
    conn: sqlite3.Connection,
    feed_url: str,
    feed_title: str | None,
    success: bool,
    status: int | None = None,
    error: str | None = None,
) -> None:
    now = utc_now_iso()
    last_successful_fetch = now if success else None

    conn.execute("""
        INSERT INTO feed_state (
            feed_url, feed_title, last_checked, last_successful_fetch, last_status, last_error
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(feed_url) DO UPDATE SET
            feed_title = excluded.feed_title,
            last_checked = excluded.last_checked,
            last_successful_fetch = COALESCE(excluded.last_successful_fetch, feed_state.last_successful_fetch),
            last_status = excluded.last_status,
            last_error = excluded.last_error
    """, (feed_url, feed_title, now, last_successful_fetch, status, error))
    conn.commit()


def upsert_feed_entries(conn: sqlite3.Connection, feed_url: str, feed) -> tuple[int, int]:
    inserted = 0
    updated = 0
    fetched_at = utc_now_iso()
    feed_title = feed.feed.get("title")

    for entry in feed.entries:
        guid = entry.get("id") or entry.get("guid")
        title = entry.get("title")
        link = entry.get("link")
        published = parse_datetime(entry)
        summary = entry.get("summary") or entry.get("description")
        author = entry.get("author")
        category = extract_category(entry)
        raw_entry = str(dict(entry))

        if not link:
            continue

        cur = conn.execute("""
            INSERT OR IGNORE INTO posts (
                feed_url, feed_title, guid, title, link, published,
                summary, author, category, fetched_at, raw_entry
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feed_url, feed_title, guid, title, link, published,
            summary, author, category, fetched_at, raw_entry
        ))

        if cur.rowcount == 1:
            inserted += 1
        else:
            cur = conn.execute("""
                UPDATE posts
                SET
                    feed_title = COALESCE(?, feed_title),
                    guid = COALESCE(?, guid),
                    title = COALESCE(?, title),
                    published = COALESCE(?, published),
                    summary = COALESCE(?, summary),
                    author = COALESCE(?, author),
                    category = COALESCE(?, category),
                    fetched_at = ?,
                    raw_entry = ?
                WHERE feed_url = ? AND link = ?
            """, (
                feed_title, guid, title, published, summary, author,
                category, fetched_at, raw_entry, feed_url, link
            ))
            if cur.rowcount == 1:
                updated += 1

    conn.commit()
    return inserted, updated


def fetch_one_feed(conn: sqlite3.Connection, feed_url: str) -> None:
    feed = feedparser.parse(feed_url)
    status = getattr(feed, "status", None)
    feed_title = feed.feed.get("title")

    if status and status >= 400:
        update_feed_state(
            conn,
            feed_url=feed_url,
            feed_title=feed_title,
            success=False,
            status=status,
            error=f"HTTP status {status}",
        )
        raise RuntimeError(f"{feed_url} failed with HTTP status {status}")

    inserted, updated = upsert_feed_entries(conn, feed_url, feed)

    update_feed_state(
        conn,
        feed_url=feed_url,
        feed_title=feed_title,
        success=True,
        status=status,
        error=None,
    )

    print(f"\nFeed: {feed_title or 'Unknown'}")
    print(f"URL: {feed_url}")
    print(f"Entries returned: {len(feed.entries)}")
    print(f"Inserted: {inserted}")
    print(f"Updated: {updated}")


def fetch_all_feeds(db_path: str = DB_PATH, feed_urls: list[str] = FEED_URLS) -> None:
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)

        for feed_url in feed_urls:
            try:
                fetch_one_feed(conn, feed_url)
            except Exception as e:
                print(f"\nError processing {feed_url}: {e}")

        print(f"\nDone. SQLite DB: {db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    fetch_all_feeds()