from __future__ import annotations

from typing import TypedDict, Any
from dataclasses import dataclass
import requests
from langgraph.graph import StateGraph, START, END
import logging

logger = logging.getLogger(__name__)
AGENT_NAME = "request_parser_agent"

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

INTEL_BASE_URL = "http://127.0.0.1:8000"

class GraphState(TypedDict):
    user_request: str
    request_type: str
    time_window_hours: int
    countries: list[str]
    sources_needed: list[str]
    retrieved_data: dict
    fused_analysis: dict
    final_report: str

# -------------------------------------------------------------------
# SIMPLE REQUEST PARSER
# -------------------------------------------------------------------

def parse_request(state: GraphState) -> GraphState:
    """
    Rule-based first pass.
    Keep this simple before adding an LLM.
    """
    user_request = state.get("user_request", "").strip()
    text = user_request.lower()

    request_type = "latest_summary"
    time_window = "last_hour"
    countries: list[str] = []
    event_types: list[str] = []
    sources_needed = ["telegram"]

    # Time window
    if "last day" in text or "today" in text:
        time_window = "last_day"
    elif "last hour" in text:
        time_window = "last_hour"

    # Request type
    if "compare" in text or "trend" in text or "last week" in text:
        request_type = "historical_comparison"
    elif "anomal" in text or "spike" in text:
        request_type = "anomaly_lookup"
    elif "what happened" in text or "summary" in text:
        request_type = "latest_summary"

    # Geography
    country_terms = [
        "iraq", "ukraine", "israel", "iran", "lebanon",
        "syria", "palestine", "bahrain", "uae"
    ]
    for term in country_terms:
        if term in text:
            countries.append(term.title())

    # Event types
    event_terms = [
        "impact", "interception", "launch", "explosion",
        "projectile", "siren"
    ]
    for term in event_terms:
        if term in text:
            event_types.append(term)

    # Sources
    if "earthquake" in text or "usgs" in text:
        sources_needed.append("earthquakes")
    if "stock" in text or "market" in text:
        sources_needed.append("stocks")

    return {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sorted(set(sources_needed)),
    }


# -------------------------------------------------------------------
# QUERY PLANNER
# -------------------------------------------------------------------

def plan_queries(state: GraphState) -> GraphState:
    """
    Decide which cached endpoints/tables to use.
    """
    request_type = state.get("request_type", "latest_summary")
    time_window = state.get("time_window", "last_hour")
    countries = state.get("countries", [])
    event_types = state.get("event_types", [])
    sources_needed = state.get("sources_needed", ["telegram"])

    use_recent_clusters = request_type in {"latest_summary", "anomaly_lookup"}
    use_hourly_anomalies = True
    use_stock_alerts = "stocks" in sources_needed
    use_earthquakes = "earthquakes" in sources_needed

    # For now, keep historical requests on cached data too.
    # Later you can branch to a custom historical analysis node.
    requires_custom_analysis = False

    retrieval_plan = {
        "cluster_window": time_window if time_window in {"last_hour", "last_day"} else "last_hour",
        "cluster_level": "country" if countries else "location",
        "countries": countries,
        "event_types": event_types,
        "use_recent_clusters": use_recent_clusters,
        "use_hourly_anomalies": use_hourly_anomalies,
        "use_stock_alerts": use_stock_alerts,
        "use_earthquakes": use_earthquakes,
        "requires_custom_analysis": requires_custom_analysis,
    }

    return {"retrieval_plan": retrieval_plan}


# -------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------

def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def retrieve_intel(state: GraphState) -> GraphState:
    """
    Pull from your FastAPI layer.
    """
    plan = state.get("retrieval_plan", {})
    countries = plan.get("countries", [])
    event_types = plan.get("event_types", [])

    cluster_window = plan.get("cluster_window", "last_hour")
    cluster_level = plan.get("cluster_level", "location")

    intel_snapshot = _get_json(
        f"{INTEL_BASE_URL}/intel/latest",
        params={
            "cluster_window": cluster_window,
            "cluster_level": cluster_level,
            "anomaly_limit": 25,
            "cluster_limit": 25,
            "stock_limit": 25,
            "earthquake_limit": 25,
        },
    )

    # Optional filtered anomaly call
    anomaly_params: dict[str, Any] = {
        "anomalies_only": "true",
        "limit": 100,
    }
    if countries:
        anomaly_params["country"] = countries[0]
    if event_types:
        anomaly_params["event_type"] = event_types[0]

    hourly_anomalies = _get_json(
        f"{INTEL_BASE_URL}/anomalies/hourly",
        params=anomaly_params,
    )

    retrieved_data = {
        "intel_snapshot": intel_snapshot,
        "hourly_anomalies": hourly_anomalies,
    }

    return {"retrieved_data": retrieved_data}


# -------------------------------------------------------------------
# FUSION / ANALYSIS
# -------------------------------------------------------------------

def fuse_findings(state: GraphState) -> GraphState:
    """
    Rule-based first pass.
    Later this is where you can plug in an LLM.
    """
    retrieved = state.get("retrieved_data", {})
    snapshot = retrieved.get("intel_snapshot", {})
    anomalies = retrieved.get("hourly_anomalies", [])
    clusters = snapshot.get("recent_clusters", [])
    stock_alerts = snapshot.get("stock_alerts", [])
    earthquakes = snapshot.get("earthquakes", [])

    top_findings: list[dict[str, Any]] = []

    for row in anomalies[:5]:
        title = f"{row.get('event_type', 'activity').title()} spike in {row.get('country', 'Unknown')}"
        evidence = []
        if row.get("spike_ratio") is not None:
            evidence.append(f"{row['spike_ratio']:.1f}x baseline")
        if row.get("current_message_count") is not None:
            evidence.append(f"{row['current_message_count']} messages")
        if row.get("current_unique_channels") is not None:
            evidence.append(f"{row['current_unique_channels']} channels")

        top_findings.append({
            "title": title,
            "confidence": "high" if row.get("is_anomaly") == 1 else "medium",
            "evidence": evidence,
            "reason": row.get("anomaly_reason"),
            "kind": "hourly_anomaly",
        })

    for row in clusters[:5]:
        top_findings.append({
            "title": f"{row.get('event_type', 'event').title()} cluster near {row.get('geo_group', 'unknown')}",
            "confidence": "medium",
            "evidence": [
                f"{row.get('message_count', 0)} messages",
                f"{row.get('channel_count', 0)} channels",
                f"severity {row.get('severity_score', 0)}",
            ],
            "reason": "recent burst cluster",
            "kind": "recent_cluster",
        })

    overall_assessment = "No major developments detected."
    if top_findings:
        overall_assessment = "Elevated activity detected across recent Telegram intelligence."

    if stock_alerts:
        overall_assessment += " Market alerts are also present."
    if earthquakes:
        overall_assessment += " Recent earthquake data is available."

    fused_analysis = {
        "top_findings": top_findings[:8],
        "overall_assessment": overall_assessment,
        "summary_counts": snapshot.get("summary_counts", {}),
    }

    return {"fused_analysis": fused_analysis}


# -------------------------------------------------------------------
# FORMATTER
# -------------------------------------------------------------------

def format_response(state: GraphState) -> GraphState:
    analysis = state.get("fused_analysis", {})
    findings = analysis.get("top_findings", [])
    overall = analysis.get("overall_assessment", "No assessment available.")

    lines = [overall, ""]

    if findings:
        lines.append("Key developments:")
        for item in findings[:5]:
            evidence = ", ".join(item.get("evidence", []))
            title = item.get("title", "Unknown finding")
            reason = item.get("reason")
            if reason:
                lines.append(f"- {title}: {evidence}. {reason}.")
            else:
                lines.append(f"- {title}: {evidence}.")
    else:
        lines.append("No significant findings in the latest snapshot.")

    final_response = "\n".join(lines)
    return {"final_response": final_response}


# -------------------------------------------------------------------
# GRAPH BUILDER
# -------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("retrieve_intel", retrieve_intel)
    graph.add_node("fuse_findings", fuse_findings)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "plan_queries")
    graph.add_edge("plan_queries", "retrieve_intel")
    graph.add_edge("retrieve_intel", "fuse_findings")
    graph.add_edge("fuse_findings", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


graph = build_graph()


if __name__ == "__main__":
    result = graph.invoke(
        {"user_request": "What happened in Israel in the last hour?"}
    )

    print("\n=== FINAL STATE ===")
    print(result)

    print("\n=== FINAL RESPONSE ===")
    print(result.get("final_response", "final_response key missing"))