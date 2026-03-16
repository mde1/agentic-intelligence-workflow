from typing import TypedDict, Any
from pydantic import BaseModel, Field

class GraphState(TypedDict):
    user_request: str
    analysis_mode: str
    request_type: str
    time_window: str                 # "last_hour" | "last_day" | "last_week"
    countries: list[str]
    event_types: list[str]
    sources_needed: list[str]
    retrieval_plan: dict[str, Any]
    retrieved_data: dict[str, Any]
    fused_analysis: dict[str, Any]
    forecast_analysis: dict[str,Any]
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
    kind: str = Field(
        ...,
        description="Type such as hourly_anomaly, recent_cluster, news_context, stock_alert, earthquake"
    )
    evidence: list[str] = Field(default_factory=list, description="Concrete evidence points")
    source_type: list[str] = Field(
        default_factory=list,
        description="Which data sources were used, such as telegram, news, markets, earthquakes"
    )
    reason: str = Field(..., description="Why this matters")

class FusedAnalysis(BaseModel):
    overall_assessment: str = Field(..., description="One-paragraph operational assessment")
    top_findings: list[Finding] = Field(default_factory=list, description="Most important findings, max 8")

class EmergingRisk(BaseModel):
    risk: str = Field(..., description="Short bullet naming a plausible emerging risk")
    rationale: str = Field(..., description="One sentence explaining why this risk could emerge if activity continues")
    watch_for: list[str] = Field(
        default_factory=list,
        description="A few concrete indicators to monitor next",
    )

class ForecastImplications(BaseModel):
    emerging_risks: list[EmergingRisk] = Field(
        default_factory=list,
        description="Two or three tightly grounded near-term risks",
    )
    confidence: str = Field(..., description="One of: high, medium, low")