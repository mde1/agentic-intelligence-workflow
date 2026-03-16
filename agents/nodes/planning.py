from agents.state import GraphState
import yaml
from pathlib import Path
import yaml
from os import PathLike
import logging
from agents.state import PlannerOutput
from agents.config import *

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# SIMPLE REQUEST PARSER
# -------------------------------------------------------------------

def load_prompt(path: str | Path | PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

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

def plan_queries(state: GraphState) -> GraphState:
    semantic_models = yaml.safe_load(SEMANTIC_MODELS_PATH.read_text(encoding="utf-8"))
    semantic_context = yaml.dump(semantic_models, sort_keys=False)

    raw_template = load_prompt(PLANNER_PROMPT_PATH)
    planner_prompt = raw_template.format(semantic_context=semantic_context)

    user_request = state.get("user_request", "")
    parsed_request_type = state.get("request_type", "latest_summary")
    parsed_time_window = state.get("time_window", "last_hour")
    parsed_countries = state.get("countries", [])
    parsed_event_types = state.get("event_types", [])

    llm = MODEL

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
    logger.info("Planner prompt loaded from %s", PLANNER_PROMPT_PATH)
    logger.info("Planner input user_request=%s", user_request)
    logger.info("Planner output=%s", plan)

    analysis_mode = plan.get("analysis_mode", "intel_summary")

    return {
        "request_type": request_type,
        "time_window": time_window,
        "countries": countries,
        "event_types": event_types,
        "sources_needed": sources_needed,
        "retrieval_plan": retrieval_plan,
        "analysis_mode": analysis_mode,
    }

def route_analysis_node(state: GraphState) -> GraphState:
    """Node: no state update; routing is done by route_analysis_edges."""
    return {}


def route_analysis_edges(state: GraphState) -> str:
    request_type = state.get("request_type", "latest_summary")

    analysis_types = {
        "earthquake_chart",
        "stock_chart",
        "data_analysis",
        "visualization",
    }

    if request_type in analysis_types:
        return "pandas_ai"

    return "intel_path"

def route_analysis(state: GraphState) -> str:
    """
    Decide whether this query should go to:
    - pandas analysis path
    - intel synthesis path
    """
    request_type = state.get("request_type", "latest_summary")

    analysis_types = {
        "earthquake_chart",
        "stock_chart",
        "data_analysis",
        "visualization",
    }

    if request_type in analysis_types:
        return "analysis"

    return "intel"