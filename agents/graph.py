from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pathlib import Path
from agents.nodes.planning import parse_request, plan_queries, route_analysis_edges, route_analysis_node
from agents.nodes.pandas_analysis import run_pandas_ai
from agents.nodes.retrieval import retrieve_intel
from agents.nodes.forecast import  forecast_implications
from agents.nodes.formatting import format_response
from agents.nodes.fusion import fuse_findings
from agents.state import GraphState
import logging
from agents.config import *

load_dotenv()

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# GRAPH BUILDER
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
        {
            "intel_path": "retrieve_intel",
            "pandas_ai": "run_pandas_ai",
        },
    )
    graph.add_edge("run_pandas_ai", END)
    graph.add_edge("retrieve_intel", "fuse_findings")
    graph.add_edge("fuse_findings", "forecast_implications")
    graph.add_edge("forecast_implications", "format_response")
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

def run_agent(user_request: str) -> str:
    result = graph.invoke({"user_request": user_request})
    return result.get("final_report", "I ran the workflow but no final report was produced.")

if __name__ == "__main__":
    #result = graph.invoke(
    #    {"user_request": "What are the latest developments"}
    #)
    save_graph_visualization()