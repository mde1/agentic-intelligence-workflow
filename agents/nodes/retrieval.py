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
    plan = state.get("retrieval_plan", {})
    countries = plan.get("countries", [])
    event_types = plan.get("event_types", [])

    retrieved = {
        "intel_snapshot": {},
        "hourly_anomalies": [],
    }

    try:
        retrieved["intel_snapshot"] = _get_json(
            f"{INTEL_BASE_URL}/intel/latest",
            params={
                "cluster_window": plan.get("cluster_window", "last_hour"),
                "cluster_level": plan.get("cluster_level", "location"),
                "anomaly_limit": 25,
                "cluster_limit": 25,
                "stock_limit": 25,
                "earthquake_limit": 25,
            },
        )
    except Exception:
        logger.exception("Failed calling /intel/latest")

    anomaly_params: dict[str, Any] = {"anomalies_only": "true", "limit": 100}
    if countries:
        anomaly_params["country"] = countries[0]
    if event_types:
        anomaly_params["event_type"] = event_types[0]

    try:
        retrieved["hourly_anomalies"] = _get_json(
            f"{INTEL_BASE_URL}/anomalies/hourly",
            params=anomaly_params,
        )
    except Exception:
        logger.exception("Failed calling /anomalies/hourly")

    if not retrieved["intel_snapshot"] and not retrieved["hourly_anomalies"]:
        return {
            "retrieved_data": retrieved,
            "final_report": "The API retrieval layer failed. Check api/routes/intel.py logs and DB paths."
        }

    return {"retrieved_data": retrieved}
