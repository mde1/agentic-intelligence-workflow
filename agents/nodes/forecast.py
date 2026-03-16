from __future__ import annotations
from agents.state import GraphState, ForecastImplications
from pathlib import Path
from os import PathLike
import logging
from agents.config import *


logger = logging.getLogger(__name__)
#AGENT_NAME = "request_parser_agent"

def load_prompt(path: str | Path | PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def forecast_implications(state: GraphState) -> GraphState:
    """
    Generate a very short near-term risk outlook.
    Keep this grounded and separate from factual fusion.
    """
    fused = state.get("fused_analysis", {}) or {}
    retrieved = state.get("retrieved_data", {}) or {}

    request_text = state.get("user_request", "")
    countries = state.get("countries", []) or []
    event_types = state.get("event_types", []) or []
    time_window = state.get("time_window", "last_hour")

    telegram = retrieved.get("telegram", {}) or {}
    news = retrieved.get("news", {}) or {}
    markets = retrieved.get("markets", {}) or {}
    earthquakes = retrieved.get("earthquakes", {}) or {}
    meta = retrieved.get("meta", {}) or {}

    payload = {
        "user_request": request_text,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_used": meta.get("sources_used", []),
        "fused_analysis": {
            "overall_assessment": fused.get("overall_assessment", ""),
            "top_findings": fused.get("top_findings", [])[:5],
        },
        "telegram": {
            "latest_hour": telegram.get("latest_hour"),
            "summary_counts": telegram.get("summary_counts", {}),
            "recent_clusters": (telegram.get("recent_clusters", []) or [])[:8],
            "hourly_anomalies": (telegram.get("hourly_anomalies", []) or [])[:8],
        },
        "news": {
            "headlines": (news.get("headlines", []) or [])[:6],
            "twz_headlines": (news.get("twz_headlines", []) or [])[:4],
        },
        "markets": {
            "stock_alerts": (markets.get("stock_alerts", []) or [])[:6],
        },
        "earthquakes": {
            "events": (earthquakes.get("events", []) or [])[:6],
        },
    }

    system_prompt = load_prompt(FORECAST_PROMPT_PATH)

    human_prompt = f"""
Assess only the most plausible near-term emerging risks if current activity continues on the present trajectory.

Rules:
- Return only 2 or 3 risks maximum.
- Each risk must be conditional and tightly grounded in the evidence.
- Do not restate the factual findings unless needed for the rationale.
- Prefer operational risks, escalation risks, spillover risks, or monitoring-relevant risks.
- If the evidence is thin or mixed, return fewer items and lower confidence.

Evidence payload:
{payload}
    """.strip()

    llm = MODEL

    structured_llm = llm.with_structured_output(
        ForecastImplications,
        method="json_schema",
    )

    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
    )

    forecast_analysis = result.model_dump()

    return {"forecast_analysis": forecast_analysis}