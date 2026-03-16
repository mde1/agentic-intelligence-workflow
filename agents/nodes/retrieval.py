from __future__ import annotations
from typing import Any
import logging
import requests
from agents.state import GraphState
from agents.config import *

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------

def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _build_telegram_params(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_window": plan.get("cluster_window", "last_hour"),
        "cluster_level": plan.get("cluster_level", "location"),
        "anomaly_limit": 25,
        "cluster_limit": 25,
    }

def fetch_telegram_recent(plan: dict[str, Any]) -> dict[str, Any]:
    params = _build_telegram_params(plan)
    snapshot = _get_json(
        f"{INTEL_BASE_URL}/intel/latest",
        params={
            **params,
            "stock_limit": 1,
            "earthquake_limit": 1,
        },
    )

    return {
        "latest_hour": snapshot.get("latest_hour"),
        "recent_clusters": snapshot.get("recent_clusters", []),
        "hourly_anomalies_snapshot": snapshot.get("hourly_anomalies", []),
        "summary_counts": {
            "recent_clusters": snapshot.get("summary_counts", {}).get("recent_clusters", 0),
            "hourly_anomalies": snapshot.get("summary_counts", {}).get("hourly_anomalies", 0),
        },
    }


def fetch_telegram_anomalies(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return _get_json(
        f"{INTEL_BASE_URL}/anomalies/hourly",
        params=_build_anomaly_params(plan),
    )


def fetch_market_alerts() -> list[dict[str, Any]]:
    snapshot = _get_json(
        f"{INTEL_BASE_URL}/intel/latest",
        params={
            "cluster_window": "last_hour",
            "cluster_level": "location",
            "anomaly_limit": 1,
            "cluster_limit": 1,
            "stock_limit": 25,
            "earthquake_limit": 1,
        },
    )
    return snapshot.get("stock_alerts", [])


def fetch_earthquakes() -> list[dict[str, Any]]:
    snapshot = _get_json(
        f"{INTEL_BASE_URL}/intel/latest",
        params={
            "cluster_window": "last_hour",
            "cluster_level": "location",
            "anomaly_limit": 1,
            "cluster_limit": 1,
            "stock_limit": 1,
            "earthquake_limit": 25,
        },
    )
    return snapshot.get("earthquakes", [])

def _build_anomaly_params(plan: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "anomalies_only": "true",
        "limit": 100,
    }

    countries = plan.get("countries", []) or []
    event_types = plan.get("event_types", []) or []

    if countries:
        params["country"] = countries[0]
    if event_types:
        params["event_type"] = event_types[0]

    return params

def retrieve_intel(state: GraphState) -> GraphState:
    plan = state.get("retrieval_plan", {}) or {}
    sources_needed = set(plan.get("sources_needed", []))

    use_recent_clusters = plan.get("use_recent_clusters", False)
    use_hourly_anomalies = plan.get("use_hourly_anomalies", False)
    use_stock_alerts = plan.get("use_stock_alerts", False)
    use_earthquakes = plan.get("use_earthquakes", False)

    telegram_data: dict[str, Any] = {}
    markets_data: dict[str, Any] = {}
    earthquakes_data: dict[str, Any] = {}

    if "telegram" in sources_needed:
        if use_recent_clusters:
            telegram_data.update(fetch_telegram_recent(plan))
        if use_hourly_anomalies:
            telegram_data["hourly_anomalies"] = fetch_telegram_anomalies(plan)

    if use_stock_alerts:
        markets_data["stock_alerts"] = fetch_market_alerts()

    if use_earthquakes:
        earthquakes_data["events"] = fetch_earthquakes()

    retrieved_data = {
        "telegram": telegram_data,
        "markets": markets_data,
        "earthquakes": earthquakes_data,
        "meta": {
            "sources_used": sorted(
                source for source, payload in {
                    "telegram": telegram_data,
                    "markets": markets_data,
                    "earthquakes": earthquakes_data,
                }.items()
                if payload
            )
        },
    }

    return {"retrieved_data": retrieved_data}
