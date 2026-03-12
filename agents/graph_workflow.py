from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class GraphState(TypedDict):
    user_request: str
    request_type: str
    time_window_hours: int
    countries: list[str]
    sources_needed: list[str]
    retrieved_data: dict
    fused_analysis: dict
    final_report: str