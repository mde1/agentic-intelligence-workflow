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



def _safe_json_loads(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    try:
        return json.loads(value)
    except Exception:
        return None


def _extract_entity_texts(entity_json, tagged_entities) -> list[str]:
    """
    Pull candidate geography strings from entity_json / tagged_entities.

    Supports a few common shapes:
    - list[dict] with keys like text/label/entity
    - dict with nested entity arrays
    - tagged_entities as comma-separated string like:
      'Kyiv (GPE), Ukraine (GPE)'
    """
    candidates = []

    parsed = _safe_json_loads(entity_json)

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                txt = (
                    item.get("text")
                    or item.get("entity")
                    or item.get("value")
                    or item.get("name")
                )
                label = (
                    item.get("label")
                    or item.get("type")
                    or item.get("entity_group")
                    or item.get("tag")
                )
                if txt and (label in {"GPE", "LOC", "FAC", "NORP"} or label is None):
                    candidates.append(str(txt))

    elif isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        txt = (
                            item.get("text")
                            or item.get("entity")
                            or item.get("value")
                            or item.get("name")
                        )
                        label = (
                            item.get("label")
                            or item.get("type")
                            or item.get("entity_group")
                            or item.get("tag")
                        )
                        if txt and (label in {"GPE", "LOC", "FAC", "NORP"} or label is None):
                            candidates.append(str(txt))

    if isinstance(tagged_entities, str) and tagged_entities.strip():
        # Example format: "Kyiv (GPE), Ukraine (GPE), IRGC (ORG)"
        pieces = [p.strip() for p in tagged_entities.split(",") if p.strip()]
        for piece in pieces:
            m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", piece)
            if m:
                txt, label = m.group(1).strip(), m.group(2).strip().upper()
                if txt and label in {"GPE", "LOC", "FAC", "NORP"}:
                    candidates.append(txt)
            else:
                candidates.append(piece)

    # de-dupe while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        c_norm = c.strip().lower()
        if c_norm and c_norm not in seen:
            seen.add(c_norm)
            deduped.append(c.strip())

    return deduped


def _match_alias_in_text(text: str):
    """
    Return the best alias match from LOCATION_ALIASES found in text.
    Uses longest-first matching so 'gaza city' beats 'gaza'.
    """
    if not text or not isinstance(text, str):
        return None

    text_lower = text.lower()
    for term in SORTED_LOCATION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return term
    return None


def resolve_geography(row: pd.Series) -> pd.Series:
    """
    Resolve geography from:
    1. entity_json
    2. tagged_entities
    3. message_text

    Returns:
    - country
    - location_key
    - location_text
    """
    message_text = row.get("message_text")
    tagged_entities = row.get("tagged_entities")
    entity_json = row.get("entity_json")

    # 1) Try extracted entities first
    entity_candidates = _extract_entity_texts(entity_json, tagged_entities)

    best_term = None
    for candidate in entity_candidates:
        matched = _match_alias_in_text(candidate)
        if matched:
            best_term = matched
            break

    # 2) Fallback to raw message text
    if best_term is None:
        best_term = _match_alias_in_text(str(message_text) if message_text is not None else "")

    # 3) Resolve
    if best_term and best_term in LOCATION_ALIASES:
        info = LOCATION_ALIASES[best_term]
        country = info.get("country")
        location_key = info.get("location_key")
        location_text = best_term
    else:
        country = None
        location_key = None
        location_text = None

    return pd.Series(
        {
            "country": country,
            "location_key": location_key,
            "location_text": location_text,
        }
    )


def build_clusters(
    events: pd.DataFrame,
    window_label: str,
    cluster_level: str = "location",   # "location" or "country"
    time_gap_seconds: int = 30,
) -> pd.DataFrame:
    """
    Build recent clusters from recent Telegram events.

    cluster_level:
        - "location": group by event_type + (location_key fallback country fallback unknown)
        - "country":  group by event_type + (country fallback unknown)

    Expected input columns:
        message_id
        channel_username
        message_date
        message_text
        tagged_entities
        entity_json
        event_type
        is_event
    """
    output_cols = [
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
    ]

    if events.empty:
        return pd.DataFrame(columns=output_cols)

    events = events.copy()

    # ---------------------------------------------------------------
    # Basic cleaning
    # ---------------------------------------------------------------
    if "is_event" in events.columns:
        events = events[events["is_event"] == 1].copy()

    events["message_date"] = pd.to_datetime(events["message_date"], utc=True, errors="coerce")
    events = events.dropna(subset=["message_date", "event_type"])

    # ---------------------------------------------------------------
    # Geography extraction inside clustering script
    # ---------------------------------------------------------------
    geo_df = events.apply(resolve_geography, axis=1)
    events = pd.concat([events, geo_df], axis=1)

    # ---------------------------------------------------------------
    # Geo grouping
    # ---------------------------------------------------------------
    if cluster_level == "location":
        # Prefer specific location, otherwise fallback to country
        events["geo_group"] = (
            events["location_key"]
            .fillna(events["country"])
            .fillna("unknown")
        )
    elif cluster_level == "country":
        events["geo_group"] = events["country"].fillna("unknown")
    else:
        raise ValueError("cluster_level must be 'location' or 'country'")

    # ---------------------------------------------------------------
    # Sort and cluster
    # ---------------------------------------------------------------
    events = events.sort_values(["event_type", "geo_group", "message_date"])

    events["time_diff_seconds"] = (
        events.groupby(["event_type", "geo_group"])["message_date"]
        .diff()
        .dt.total_seconds()
    )

    events["new_cluster"] = (
        events["time_diff_seconds"].isna()
        | (events["time_diff_seconds"] > time_gap_seconds)
    )

    events["cluster_num"] = (
        events.groupby(["event_type", "geo_group"])["new_cluster"]
        .cumsum()
    )

    events["event_cluster_id"] = (
        events["event_type"].astype(str)
        + "_"
        + events["geo_group"].astype(str)
        + "_"
        + events["cluster_num"].astype(str)
    )

    # ---------------------------------------------------------------
    # Aggregate cluster summary
    # ---------------------------------------------------------------
    cluster_summary = (
        events.groupby(["event_cluster_id", "event_type", "geo_group"], as_index=False)
        .agg(
            cluster_start=("message_date", "min"),
            cluster_end=("message_date", "max"),
            message_count=("message_id", "count"),
            channels=("channel_username", lambda x: ", ".join(sorted(set(v for v in x.dropna() if v)))),
            channel_count=("channel_username", "nunique"),
            sample_text=("message_text", "first"),
            entities=("tagged_entities", lambda x: ", ".join(sorted(set(v for v in x.dropna() if v)))),
            country=("country", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None),
            location_key=("location_key", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None),
            location_text=("location_text", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None),
        )
    )

    cluster_summary["duration_seconds"] = (
        cluster_summary["cluster_end"] - cluster_summary["cluster_start"]
    ).dt.total_seconds()

    cluster_summary["severity_score"] = (
        cluster_summary["message_count"] + 2 * cluster_summary["channel_count"]
    )

    cluster_summary["window_label"] = window_label
    cluster_summary["cluster_level"] = cluster_level
    cluster_summary["computed_at"] = datetime.now(timezone.utc).isoformat()

    cluster_summary = cluster_summary.sort_values(
        ["severity_score", "cluster_start"],
        ascending=[False, False]
    )

    return cluster_summary[output_cols]