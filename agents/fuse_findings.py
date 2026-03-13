from typing import Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Finding(BaseModel):
    title: str = Field(..., description="Short analyst-style headline")
    confidence: str = Field(..., description="One of: high, medium, low")
    kind: str = Field(..., description="Type such as hourly_anomaly, recent_cluster, stock_alert, earthquake")
    evidence: list[str] = Field(default_factory=list, description="Concrete evidence points")
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

        Rules:
        - Prefer developments supported by multiple indicators
        - If evidence is weak, say so
        - Do not invent facts not present in the input
        - Focus on operational significance, not generic summaries
        - Return no more than 8 findings
        - Confidence must be one of: high, medium, low
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