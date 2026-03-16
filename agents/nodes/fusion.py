from __future__ import annotations
from typing import Any
import logging
from pathlib import Path
from os import PathLike
from agents.state import GraphState, FusedAnalysis
from agents.config import *

logger = logging.getLogger(__name__)


def load_prompt(path: str | Path | PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _limit_items(items: list[dict[str, Any]] | None, n: int) -> list[dict[str, Any]]:
    return (items or [])[:n]

def fuse_findings(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved_data", {}) or {}

    telegram = retrieved.get("telegram", {}) or {}
    news = retrieved.get("news", {}) or {}
    markets = retrieved.get("markets", {}) or {}
    earthquakes = retrieved.get("earthquakes", {}) or {}
    meta = retrieved.get("meta", {}) or {}

    request_text = state.get("user_request", "")
    countries = state.get("countries", [])
    event_types = state.get("event_types", [])
    time_window = state.get("time_window", "last_hour")
    request_type = state.get("request_type", "latest_summary")
    sources_needed = state.get("sources_needed", [])

    telegram_payload = {
        "latest_hour": telegram.get("latest_hour"),
        "summary_counts": telegram.get("summary_counts", {}),
        "recent_clusters": (telegram.get("recent_clusters", []) or [])[:15],
        "hourly_anomalies": (telegram.get("hourly_anomalies", []) or [])[:15],
    }

    news_payload = {
        "headlines": (news.get("headlines", []) or [])[:12],
        "twz_headlines": (news.get("twz_headlines", []) or [])[:8],
        "summary_counts": news.get("summary_counts", {}),
    }

    markets_payload = {
        "stock_alerts": (markets.get("stock_alerts", []) or [])[:10],
        "summary_counts": markets.get("summary_counts", {}),
    }

    earthquakes_payload = {
        "events": (earthquakes.get("events", []) or [])[:10],
        "summary_counts": earthquakes.get("summary_counts", {}),
    }

    payload = {
        "user_request": request_text,
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sources_needed,
        "sources_used": meta.get("sources_used", []),
        "telegram": telegram_payload,
        "news": news_payload,
        "markets": markets_payload,
        "earthquakes": earthquakes_payload,
    }

    system_prompt = load_prompt(FUSION_PROMPT_PATH)

    human_prompt = f"""
Analyze this intelligence snapshot and return a structured fused assessment.

Rules:
- Prioritize signal over noise.
- Prefer cross-source corroboration when available.
- Use concrete metrics or timestamps where present.
- If a source category is empty, ignore it rather than speculating.
- Focus on operationally important developments.

Snapshot:
{payload}
    """.strip()

    llm = MODEL

    structured_llm = llm.with_structured_output(
        FusedAnalysis,
        method="json_schema",
    )

    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
    )

    fused_analysis = result.model_dump()

    return {"fused_analysis": fused_analysis}