from __future__ import annotations
from typing import TypedDict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
import requests
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
#AGENT_NAME = "request_parser_agent"

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

class Finding(BaseModel):
    title: str = Field(..., description="Short analyst-style headline")
    confidence: str = Field(..., description="One of: high, medium, low")
    kind: str = Field(..., description="Type such as hourly_anomaly, recent_cluster, stock_alert, earthquake")
    evidence: list[str] = Field(default_factory=list, description="Concrete evidence points")
    source_type: list[str] = Field(default_factory=list, description="Which data sources were these observations gleaned from")
    reason: str = Field(..., description="Why this matters")


class FusedAnalysis(BaseModel):
    overall_assessment: str = Field(..., description="One-paragraph operational assessment")
    top_findings: list[Finding] = Field(default_factory=list, description="Most important findings, max 8")


def fuse_findings(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved_data", {})
    snapshot = retrieved.get("intel_snapshot", {})
    hourly_anomalies = retrieved.get("hourly_anomalies", [])

    request_text = state.get("user_request", "")
    countries = state.get("countries", [])
    event_types = state.get("event_types", [])
    time_window = state.get("time_window", "last_hour")

    # Keep raw data in state; prompt with only the parts the model needs.
    payload = {
        "user_request": request_text,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "summary_counts": snapshot.get("summary_counts", {}),
        "hourly_anomalies": hourly_anomalies[:15],
        "recent_clusters": snapshot.get("recent_clusters", [])[:15],
        "stock_alerts": snapshot.get("stock_alerts", [])[:10],
        "earthquakes": snapshot.get("earthquakes", [])[:10],
    }

    system_prompt = """
        You are a threat intelligence fusion geopolitical analyst for a real-time OSINT monitoring platform.

        Your job:
        - identify the most important developments
        - prioritize signal over noise
        - use concrete metrics from the input
        - avoid speculation
        - clearly state uncertainty
        - produce concise analyst-style findings
        - group findings by region

        Rules:
        - Prefer developments supported by multiple indicators
        - If evidence is weak, say so
        - Do not invent facts not present in the input
        - Focus on operational significance, not generic summaries
        - Return no more than 15 findings
        - Confidence must be one of: high, medium, low
        - Compare the current developments with a summary of the last several hours
        """

    human_prompt = f"""
        Analyze this latest intelligence snapshot and return a structured fused assessment.

        Snapshot:
        {payload}
        """

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

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


# -------------------------------------------------------------------
# FORMATTER
# -------------------------------------------------------------------

def format_response(state: GraphState) -> GraphState:
    print(">>> RUNNING format_response")
    analysis = state.get("fused_analysis", {}) or {}
    findings = analysis.get("top_findings", []) or []
    overall = analysis.get("overall_assessment", "No major developments detected.")

    parts = [overall.strip()]

    if findings:
        bullets = []
        for item in findings[:5]:
            title = item.get("title", "Unnamed finding")
            confidence = item.get("confidence", "unknown").upper()
            evidence = item.get("evidence", []) or []
            source_type = item.get("source_type", []) or []
            reason = item.get("reason", "")

            evidence_text = ", ".join(str(x) for x in evidence if x)

            bullet = f"• {title} ({confidence})"
            if evidence_text:
                bullet += f" — {evidence_text}"
            if reason:
                bullet += f". {reason}"
            if source_type:
                bullet += f" - {source_type}"

            bullets.append(bullet)

        parts.append("\nKey developments:\n" + "\n".join(bullets))
    else:
        parts.append("\nNo significant findings in the latest snapshot.")

    final_report = "\n".join(parts).strip()

    print(">>> RETURNING final_response")
    print(final_report)

    return {"final_report": final_report}


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

# ---------------------------
# 6. Visualize Graph
# ---------------------------

def save_graph_visualization():
    PROJECT_ROOT = Path(__file__).parent.parent
    """Save the workflow graph as a PNG image."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = PROJECT_ROOT / "agents" / "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Workflow graph saved to outputs/workflow_graph.png\n")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}\n")


if __name__ == "__main__":
    result = graph.invoke(
        {"user_request": "What is the last timestamp in today's messages?"}
    )
    save_graph_visualization()
    #print("\n=== RESULT TYPE ===")
    #print(type(result))

    #print("\n=== FINAL STATE KEYS ===")
    #print(list(result.keys()))

    #print("\n=== FULL RESULT ===")
    #print(result)

    print("\n=== FINAL RESPONSE ===")
    print(result.get("final_report", "final_report key missing"))