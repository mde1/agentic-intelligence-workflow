from __future__ import annotations
from typing import Any
from agents.nodes.retrieval import retrieve_intel
from agents.state import GraphState
from agents.config import *

def build_dataframe(retrieved: dict[str, Any]):
    """Build a pandas DataFrame from normalized retrieved_data."""
    import pandas as pd

    rows: list[dict[str, Any]] = []

    telegram = retrieved.get("telegram", {}) or {}
    markets = retrieved.get("markets", {}) or {}
    earthquakes = retrieved.get("earthquakes", {}) or {}

    for cluster in (telegram.get("recent_clusters") or [])[:100]:
        rows.append(
            {
                "source": "telegram_cluster",
                "country": cluster.get("country", ""),
                "event_type": cluster.get("event_type", ""),
                "severity": cluster.get("severity_score"),
                "ratio": None,
                "message_count": cluster.get("message_count"),
                "location": cluster.get("location_text") or cluster.get("location_key", ""),
                "channels": cluster.get("channel_count"),
                "timestamp": cluster.get("cluster_start"),
                "title": None,
                "symbol": None,
                "magnitude": None,
                "depth_km": None,
            }
        )

    for item in (telegram.get("hourly_anomalies") or [])[:100]:
        rows.append(
            {
                "source": "telegram_anomaly",
                "country": item.get("country", ""),
                "event_type": item.get("event_type", ""),
                "severity": item.get("zscore_messages"),
                "ratio": item.get("spike_ratio"),
                "message_count": item.get("current_message_count") or item.get("message_count"),
                "location": item.get("location", ""),
                "channels": item.get("current_unique_channels"),
                "timestamp": item.get("hour_bucket") or item.get("computed_at"),
                "title": None,
                "symbol": None,
                "magnitude": None,
                "depth_km": None,
            }
        )

    for item in (markets.get("stock_alerts") or [])[:100]:
        rows.append(
            {
                "source": "stock_alert",
                "country": None,
                "event_type": "stock_alert",
                "severity": item.get("z_score") or item.get("severity"),
                "ratio": item.get("pct_change") or item.get("ratio"),
                "message_count": None,
                "location": None,
                "channels": None,
                "timestamp": item.get("date") or item.get("timestamp"),
                "title": item.get("company") or item.get("headline"),
                "symbol": item.get("symbol"),
                "magnitude": None,
                "depth_km": None,
            }
        )

    for item in (earthquakes.get("events") or [])[:100]:
        rows.append(
            {
                "source": "earthquake",
                "country": None,
                "event_type": "earthquake",
                "severity": item.get("magnitude"),
                "ratio": None,
                "message_count": None,
                "location": item.get("location"),
                "channels": None,
                "timestamp": item.get("time"),
                "title": item.get("title"),
                "symbol": None,
                "magnitude": item.get("magnitude"),
                "depth_km": item.get("depth_km"),
            }
        )

    columns = [
        "source",
        "country",
        "event_type",
        "severity",
        "ratio",
        "message_count",
        "location",
        "channels",
        "timestamp",
        "title",
        "symbol",
        "magnitude",
        "depth_km",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def run_pandas_ai(state: GraphState) -> GraphState:
    query = state["user_request"]
    retrieved = state.get("retrieved_data", {}) or {}
    plan = state.get("retrieval_plan", {}) or {}

    # If we routed to pandas without retrieval, fetch using the new normalized path.
    has_telegram = bool((retrieved.get("telegram", {}) or {}).get("recent_clusters") or
                        (retrieved.get("telegram", {}) or {}).get("hourly_anomalies"))
    has_markets = bool((retrieved.get("markets", {}) or {}).get("stock_alerts"))
    has_earthquakes = bool((retrieved.get("earthquakes", {}) or {}).get("events"))

    if not retrieved or not (has_telegram or has_markets or has_earthquakes):
        try:
            retrieval_update = retrieve_intel(state)
            retrieved = retrieval_update.get("retrieved_data", {}) or {}
        except Exception as e:
            logger.warning("PandasAI path: could not fetch normalized intel: %s", e)

    df = build_dataframe(retrieved)

    if df.empty:
        return {"final_report": "No data was available for analysis."}

    try:
        import os
        import matplotlib
        matplotlib.use("Agg")

        from pandasai import Agent
        from pandasai_openai import OpenAI as PAIOpenAI

        llm = PAIOpenAI(api_token=os.getenv("OPENAI_API_KEY"))

        plotly_instruction = """
You are analyzing a pandas DataFrame containing normalized intelligence data.

Rules:
- Use Plotly only for charts.
- Do NOT use matplotlib.
- Do NOT use seaborn.
- Prefer plotly.express unless lower-level Plotly is clearly needed.
- If the user asks for a chart, graph, plot, or visualization, create it with Plotly.
- Return a concise written summary along with the result.
- If you generate a chart, save it instead of trying to display it inline.
- The 'source' column distinguishes telegram clusters, telegram anomalies, stock alerts, and earthquakes.
- Do not assume every column is populated for every source type.
"""

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

    except Exception as e:
        logger.exception("PandasAI run failed: %s", e)
        result_str = f"Data analysis could not be completed: {e}"

    return {"final_report": result_str}
