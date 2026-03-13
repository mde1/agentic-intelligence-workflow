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
import yaml

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
    time_window: str                 # "last_hour" | "last_day" | "last_week"
    countries: list[str]
    event_types: list[str]
    sources_needed: list[str]
    retrieval_plan: dict[str, Any]
    retrieved_data: dict[str, Any]
    fused_analysis: dict[str, Any]
    final_report: str

# -------------------------------------------------------------------
# SIMPLE REQUEST PARSER
# -------------------------------------------------------------------

def parse_request(state: GraphState) -> GraphState:
    user_request = state.get("user_request", "").strip()
    text = user_request.lower()

    request_type = "latest_summary"
    time_window = "last_hour"
    countries = []
    event_types = []

    if any(x in text for x in ["last day", "today"]):
        time_window = "last_day"
    elif "last week" in text:
        time_window = "last_week"

    if any(x in text for x in ["compare", "trend", "historical", "last week"]):
        request_type = "historical_comparison"
    elif any(x in text for x in ["anomal", "spike", "surge"]):
        request_type = "anomaly_lookup"
    elif any(x in text for x in ["last timestamp", "latest timestamp", "most recent message"]):
        request_type = "metadata_lookup"

    country_terms = ["iraq", "ukraine", "israel", "iran", "lebanon", "syria", "palestine", "bahrain", "uae"]
    countries = [term.title() for term in country_terms if term in text]

    event_terms = ["impact", "interception", "launch", "explosion", "projectile", "siren"]
    event_types = [term for term in event_terms if term in text]

    return {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
    }


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "planner_prompt.txt"
SEMANTIC_MODELS_PATH = PROJECT_ROOT / "semantic" / "semantic_models.yml"

def load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
# -------------------------------------------------------------------
# QUERY PLANNER
# -------------------------------------------------------------------

class PlannerOutput(BaseModel):
    request_type: str
    time_window: str
    countries: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    sources_needed: list[str] = Field(default_factory=list)

def plan_queries(state: GraphState) -> GraphState:
    semantic_models = yaml.safe_load(SEMANTIC_MODELS_PATH.read_text(encoding="utf-8"))
    semantic_context = yaml.dump(semantic_models, sort_keys=False)

    raw_template = load_prompt(PROMPT_PATH)
    planner_prompt = raw_template.format(semantic_context=semantic_context)

    user_request = state.get("user_request", "")
    parsed_request_type = state.get("request_type", "latest_summary")
    parsed_time_window = state.get("time_window", "last_hour")
    parsed_countries = state.get("countries", [])
    parsed_event_types = state.get("event_types", [])

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    structured_llm = llm.with_structured_output(PlannerOutput, method="json_schema")

    planner_input = f"""
User request: {user_request}

Current parser hints:
- request_type: {parsed_request_type}
- time_window: {parsed_time_window}
- countries: {parsed_countries}
- event_types: {parsed_event_types}

Return the best retrieval plan JSON.
""".strip()

    plan = structured_llm.invoke([
        {"role": "system", "content": planner_prompt},
        {"role": "user", "content": planner_input},
    ]).model_dump()

    request_type = plan["request_type"]
    time_window = plan["time_window"]
    countries = plan.get("countries", [])
    event_types = plan.get("event_types", [])
    sources_needed = plan.get("sources_needed", [])

    retrieval_plan = {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sources_needed,
        "cluster_window": time_window if time_window in {"last_hour", "last_day", "last_week"} else "last_hour",
        "cluster_level": "country" if countries else "location",
        "use_recent_clusters": request_type in {"latest_summary", "anomaly_lookup", "historical_comparison"},
        "use_hourly_anomalies": request_type in {"latest_summary", "anomaly_lookup", "historical_comparison"},
        "use_stock_alerts": "stocks" in sources_needed,
        "use_earthquakes": "earthquakes" in sources_needed,
        "requires_custom_analysis": request_type == "historical_comparison",
    }

    return {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sources_needed,
        "retrieval_plan": retrieval_plan,
    }


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
        - add a final paragraph speculating vulnerabilities or chain reactions that could occur as a result of the latest developments.

        Rules:
        - Prefer developments supported by multiple indicators
        - If evidence is weak, say so
        - Do not invent facts not present in the input
        - Focus on operational significance, not generic summaries
        - Return no more than 15 findings
        - Confidence must be one of: high, medium, low
        - Compare the current developments with a summary of the last several hours
        - Specify the latest timestamp of the message_date
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
        {"user_request": "What's happening today?"}
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