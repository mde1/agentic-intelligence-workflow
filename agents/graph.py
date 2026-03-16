from __future__ import annotations
from typing import Any, TypedDict
import logging
import os
import sqlite3
from pathlib import Path
from os import PathLike
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI


load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODELS = {
    "openai": ChatOpenAI(model="gpt-5.4-2026-03-05", temperature=0),
    "grok": ChatXAI(model="grok-4-1-fast-reasoning", temperature=0),
    "anthropic": ChatAnthropic(model="claude-opus-4-6", temperature=0),
    "google": ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0),
}
MODEL = LLM_MODELS["openai"]

INTEL_BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANNER_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "planner_prompt.txt"
SEMANTIC_MODELS_PATH = PROJECT_ROOT / "semantic" / "semantic_models.yml"
FORECAST_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "forecast_implications.txt"
FUSION_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "fusion_prompt.txt"


class GraphState(TypedDict, total=False):
    user_request: str
    analysis_mode: str
    request_type: str
    time_window: str
    countries: list[str]
    event_types: list[str]
    sources_needed: list[str]
    retrieval_plan: dict[str, Any]
    retrieved_data: dict[str, Any]
    analysis_dataframe_info: dict[str, Any]
    fused_analysis: dict[str, Any]
    forecast_analysis: dict[str, Any]
    final_report: str


class PlannerOutput(BaseModel):
    request_type: str
    time_window: str
    countries: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    sources_needed: list[str] = Field(default_factory=list)
    analysis_mode: str


class Finding(BaseModel):
    title: str = Field(..., description="Short analyst-style headline")
    confidence: str = Field(..., description="One of: high, medium, low")
    kind: str = Field(..., description="Type such as hourly_anomaly, recent_cluster, stock_alert, earthquake")
    evidence: list[str] = Field(default_factory=list)
    source_type: list[str] = Field(default_factory=list)
    reason: str = Field(..., description="Why this matters")


class FusedAnalysis(BaseModel):
    overall_assessment: str = Field(..., description="One-paragraph operational assessment")
    top_findings: list[Finding] = Field(default_factory=list, description="Most important findings, max 8")


class ForecastItem(BaseModel):
    scenario: str = Field(..., description="Short statement of a plausible near-term development")
    likelihood: str = Field(..., description="One of: high, medium, low")
    rationale: str = Field(..., description="Why this scenario is plausible based on the evidence")
    indicators: list[str] = Field(default_factory=list, description="Specific indicators to monitor")


class ForecastImplications(BaseModel):
    outlook: str
    plausible_chain_reactions: list[ForecastItem] = Field(default_factory=list)
    vulnerabilities: list[str] = Field(default_factory=list)
    monitoring_priorities: list[str] = Field(default_factory=list)
    confidence: str


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def load_prompt(path: str | Path | PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_semantic_models() -> dict[str, Any]:
    with open(SEMANTIC_MODELS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def normalize_db_path(raw_path: str) -> Path:
    return PROJECT_ROOT / raw_path


def choose_analysis_table(state: GraphState) -> str:
    text = state.get("user_request", "").lower()
    sources_needed = set(state.get("sources_needed", []) or [])
    request_type = state.get("request_type", "latest_summary")

    if "earthquakes" in sources_needed or any(k in text for k in ["earthquake", "earthquakes", "seismic", "magnitude", "quake", "quakes", "epicenter"]):
        return "usgs_earthquakes"

    if "stocks" in sources_needed:
        if any(k in text for k in ["alert", "alerts", "warning", "warnings", "spike", "drop", "decline"]):
            return "stock_alerts"
        return "stocks_data_long"

    if request_type == "anomaly_lookup" or any(k in text for k in ["anomaly", "spike", "surge", "above baseline"]):
        return "telegram_hourly_anomalies"

    return "telegram_cluster_metrics_recent"


def _coerce_time_filter(df: pd.DataFrame, column: str, time_window: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    working = df.copy()
    working[column] = pd.to_datetime(working[column], errors="coerce", utc=True)
    working = working.dropna(subset=[column])

    if working.empty:
        return working

    latest = working[column].max()
    if pd.isna(latest):
        return working

    if time_window == "last_hour":
        cutoff = latest - pd.Timedelta(hours=1)
    elif time_window == "last_day":
        cutoff = latest - pd.Timedelta(days=1)
    elif time_window == "last_week":
        cutoff = latest - pd.Timedelta(days=7)
    else:
        return working

    return working.loc[working[column] >= cutoff].copy()


def load_analysis_dataframe(state: GraphState) -> tuple[pd.DataFrame, dict[str, Any]]:
    semantic_models = load_semantic_models()
    table_key = choose_analysis_table(state)
    model = semantic_models.get(table_key)
    if not model:
        raise ValueError(f"No semantic model found for analysis table '{table_key}'")

    data_source = model.get("data_source", {})
    db_path = normalize_db_path(data_source["database"])
    table_name = data_source["table"]
    time_field = model.get("primary_time_field")
    time_window = state.get("time_window", "last_hour")
    countries = set(state.get("countries", []) or [])
    event_types = set(state.get("event_types", []) or [])

    query = f'SELECT * FROM "{table_name}" ORDER BY "{time_field}" DESC LIMIT 5000' if time_field else f'SELECT * FROM "{table_name}" LIMIT 5000'

    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if time_field:
        df = _coerce_time_filter(df, time_field, time_window)

    if countries and "country" in df.columns:
        df = df[df["country"].isin(countries)]

    if event_types and "event_type" in df.columns:
        df = df[df["event_type"].isin(event_types)]

    info = {
        "table_key": table_key,
        "table_name": table_name,
        "database": str(db_path),
        "row_count": len(df),
        "time_field": time_field,
        "columns": list(df.columns),
    }
    return df, info


# -------------------------------------------------------------------
# parser / planner
# -------------------------------------------------------------------

def parse_request(state: GraphState) -> GraphState:
    user_request = state.get("user_request", "").strip()
    text = user_request.lower()

    request_type = "latest_summary"
    time_window = "last_hour"

    if any(x in text for x in ["last day", "today", "24 hours"]):
        time_window = "last_day"
    elif any(x in text for x in ["last week", "7 days"]):
        time_window = "last_week"

    if any(x in text for x in ["compare", "trend", "historical", "over time"]):
        request_type = "historical_comparison"
    elif any(x in text for x in ["anomal", "spike", "surge", "above baseline"]):
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


def plan_queries(state: GraphState) -> GraphState:
    semantic_models = load_semantic_models()
    semantic_context = yaml.dump(semantic_models, sort_keys=False)
    planner_prompt = load_prompt(PLANNER_PROMPT_PATH).format(semantic_context=semantic_context)

    planner_input = f"""
    User request: {state.get('user_request', '')}

    Current parser hints:
    - request_type: {state.get('request_type', 'latest_summary')}
    - time_window: {state.get('time_window', 'last_hour')}
    - countries: {state.get('countries', [])}
    - event_types: {state.get('event_types', [])}

    Return the best retrieval plan JSON.
    """.strip()

    structured_llm = MODEL.with_structured_output(PlannerOutput, method="json_schema")
    plan = structured_llm.invoke(
        [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": planner_input},
        ]
    ).model_dump()

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

    logger.info("Planner output=%s", plan)
    return {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sources_needed,
        "retrieval_plan": retrieval_plan,
        "analysis_mode": plan.get("analysis_mode", "intel_summary"),
    }


# -------------------------------------------------------------------
# routing / retrieval
# -------------------------------------------------------------------

def route_analysis_node(state: GraphState) -> GraphState:
    return {}


def route_analysis_edges(state: GraphState) -> str:
    mode = state.get("analysis_mode", "intel_summary")
    if mode in ["data_analysis", "visualization"]:
        return "pandas_ai"
    return "intel_path"


def retrieve_intel(state: GraphState) -> GraphState:
    plan = state.get("retrieval_plan", {})
    countries = plan.get("countries", [])
    event_types = plan.get("event_types", [])

    intel_snapshot = _get_json(
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

    anomaly_params: dict[str, Any] = {"anomalies_only": "true", "limit": 100}
    if countries:
        anomaly_params["country"] = countries[0]
    if event_types:
        anomaly_params["event_type"] = event_types[0]

    hourly_anomalies = _get_json(f"{INTEL_BASE_URL}/anomalies/hourly", params=anomaly_params)
    return {"retrieved_data": {"intel_snapshot": intel_snapshot, "hourly_anomalies": hourly_anomalies}}


def build_snapshot_dataframe(retrieved: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot = retrieved.get("intel_snapshot", {}) or {}

    for cluster in snapshot.get("recent_clusters", [])[:100]:
        rows.append(
            {
                "source": "cluster",
                "country": cluster.get("country", ""),
                "event_type": cluster.get("event_type", ""),
                "severity_score": cluster.get("severity_score"),
                "message_count": cluster.get("message_count"),
                "location_text": cluster.get("location_text", ""),
                "cluster_start": cluster.get("cluster_start"),
            }
        )

    for item in (retrieved.get("hourly_anomalies") or [])[:100]:
        rows.append(
            {
                "source": "anomaly",
                "country": item.get("country", ""),
                "event_type": item.get("event_type", ""),
                "spike_ratio": item.get("spike_ratio"),
                "current_message_count": item.get("current_message_count"),
                "hour_bucket": item.get("hour_bucket"),
                "anomaly_reason": item.get("anomaly_reason", ""),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# pandas analysis
# -------------------------------------------------------------------

def run_pandas_ai(state: GraphState) -> GraphState:
    query = state["user_request"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        from pandasai import Agent
        from pandasai_openai import OpenAI as PAIOpenAI

        df, df_info = load_analysis_dataframe(state)
        if df.empty:
            return {
                "final_report": (
                    f"The selected analysis source '{df_info['table_name']}' loaded successfully, but it returned 0 rows "
                    f"after applying the current filters/time window. Database: {df_info['database']}"
                ),
                "analysis_dataframe_info": df_info,
            }

        llm = PAIOpenAI(api_token=os.getenv("OPENAI_API_KEY"))
        plotly_instruction = f"""
        You are analyzing a pandas DataFrame loaded directly from SQLite.

        Data source details:
        - semantic table key: {df_info['table_key']}
        - physical table name: {df_info['table_name']}
        - database: {df_info['database']}
        - row_count: {df_info['row_count']}
        - columns: {df_info['columns']}

        Rules:
        - Use Plotly only.
        - Do NOT use matplotlib.
        - Do NOT use seaborn.
        - If the user asks for a chart, graph, or plot, create it with Plotly.
        - Use the actual dataframe columns listed above.
        - Do not invent table names or SQL queries.
        - Work only with the dataframe already provided.
        - Return a concise written summary along with the result.
        - If you generate a chart, save it instead of trying to display it inline.
        """.strip()

        agent = Agent(
            df,
            config={
                "llm": llm,
                "save_charts": True,
                "verbose": True,
            },
        )
        result = agent.chat(f"{plotly_instruction}\n\nUser request: {query}")
        result_str = str(result) if result is not None else "No result from analysis."
        return {"final_report": result_str, "analysis_dataframe_info": df_info}
    except Exception as e:
        logger.exception("PandasAI run failed: %s", e)

        fallback = state.get("retrieved_data", {})
        if fallback:
            fallback_df = build_snapshot_dataframe(fallback)
            if not fallback_df.empty:
                return {
                    "final_report": f"Data analysis could not be completed with the direct table loader: {e}",
                    "analysis_dataframe_info": {
                        "table_key": "snapshot_fallback",
                        "row_count": len(fallback_df),
                        "columns": list(fallback_df.columns),
                    },
                }

        return {"final_report": f"Data analysis could not be completed: {e}"}


# -------------------------------------------------------------------
# fusion / forecast / formatting
# -------------------------------------------------------------------

def fuse_findings(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved_data", {})
    snapshot = retrieved.get("intel_snapshot", {})
    hourly_anomalies = retrieved.get("hourly_anomalies", [])

    payload = {
        "user_request": state.get("user_request", ""),
        "time_window": state.get("time_window", "last_hour"),
        "countries": state.get("countries", []),
        "event_types": state.get("event_types", []),
        "summary_counts": snapshot.get("summary_counts", {}),
        "hourly_anomalies": hourly_anomalies[:15],
        "recent_clusters": snapshot.get("recent_clusters", [])[:15],
        "stock_alerts": snapshot.get("stock_alerts", [])[:10],
        "earthquakes": snapshot.get("earthquakes", [])[:10],
    }

    structured_llm = MODEL.with_structured_output(FusedAnalysis, method="json_schema")
    result = structured_llm.invoke(
        [
            {"role": "system", "content": load_prompt(FUSION_PROMPT_PATH)},
            {"role": "user", "content": f"Analyze this latest intelligence snapshot and return a structured fused assessment.\n\nSnapshot:\n{payload}"},
        ]
    )
    return {"fused_analysis": result.model_dump()}


def forecast_implications(state: GraphState) -> GraphState:
    fused = state.get("fused_analysis", {}) or {}
    retrieved = state.get("retrieved_data", {}) or {}
    snapshot = retrieved.get("intel_snapshot", {}) or {}
    hourly_anomalies = retrieved.get("hourly_anomalies", []) or []

    payload = {
        "user_request": state.get("user_request", ""),
        "time_window": state.get("time_window", "last_hour"),
        "countries": state.get("countries", []),
        "event_types": state.get("event_types", []),
        "fused_analysis": {
            "overall_assessment": fused.get("overall_assessment", ""),
            "top_findings": fused.get("top_findings", [])[:6],
        },
        "summary_counts": snapshot.get("summary_counts", {}),
        "recent_clusters": snapshot.get("recent_clusters", [])[:12],
        "hourly_anomalies": hourly_anomalies[:12],
        "stock_alerts": snapshot.get("stock_alerts", [])[:8],
        "earthquakes": snapshot.get("earthquakes", [])[:8],
    }

    structured_llm = MODEL.with_structured_output(ForecastImplications, method="json_schema")
    result = structured_llm.invoke(
        [
            {"role": "system", "content": load_prompt(FORECAST_PROMPT_PATH)},
            {"role": "user", "content": f"Assess the near-term implications of this intelligence picture over the next few days.\n\nEvidence payload:\n{payload}"},
        ]
    )
    return {"forecast_analysis": result.model_dump()}


def format_response(state: GraphState) -> GraphState:
    analysis = state.get("fused_analysis", {}) or {}
    forecast = state.get("forecast_analysis", {}) or {}

    findings = analysis.get("top_findings", []) or []
    overall = analysis.get("overall_assessment", "No major developments detected.")
    parts = [overall.strip()]

    if findings:
        bullets = []
        for item in findings[:5]:
            bullet = f"• {item.get('title', 'Unnamed finding')} ({item.get('confidence', 'unknown').upper()})"
            evidence = ", ".join(str(x) for x in (item.get("evidence", []) or []) if x)
            if evidence:
                bullet += f" — {evidence}"
            if item.get("reason"):
                bullet += f". {item['reason']}"
            bullets.append(bullet)
        parts.append("\nKey developments:\n" + "\n".join(bullets))

    if forecast:
        forecast_parts = []
        if forecast.get("outlook"):
            forecast_parts.append(forecast["outlook"])
        if forecast.get("monitoring_priorities"):
            forecast_parts.append(
                "Monitoring priorities:\n" + "\n".join(f"• {x}" for x in forecast.get("monitoring_priorities", [])[:5])
            )
        forecast_parts.append(f"Assessment confidence: {forecast.get('confidence', 'unknown').upper()}")
        parts.append("\nNear-term implications:\n" + "\n\n".join(forecast_parts))

    return {"final_report": "\n".join(parts).strip()}


# -------------------------------------------------------------------
# graph builder
# -------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("route_analysis", route_analysis_node)
    graph.add_node("retrieve_intel", retrieve_intel)
    graph.add_node("run_pandas_ai", run_pandas_ai)
    graph.add_node("fuse_findings", fuse_findings)
    graph.add_node("forecast_implications", forecast_implications)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "plan_queries")
    graph.add_edge("plan_queries", "route_analysis")
    graph.add_conditional_edges(
        "route_analysis",
        route_analysis_edges,
        {"intel_path": "retrieve_intel", "pandas_ai": "run_pandas_ai"},
    )
    graph.add_edge("run_pandas_ai", END)
    graph.add_edge("retrieve_intel", "fuse_findings")
    graph.add_edge("fuse_findings", "forecast_implications")
    graph.add_edge("forecast_implications", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


graph = build_graph()


def save_graph_visualization():
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = PROJECT_ROOT / "agents" / "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
    except Exception as e:
        logger.warning("Could not generate graph visualization: %s", e)


def run_agent(user_request: str) -> str:
    result = graph.invoke({"user_request": user_request})
    return result.get("final_report", "I ran the workflow but no final report was produced.")


if __name__ == "__main__":
    save_graph_visualization()
