# Agents Architecture

## OSINT Intelligence Platform

This document describes the **agent architecture** used in the OSINT
Intelligence Monitoring System.

The platform aggregates OSINT signals (Telegram activity, news feeds,
seismic events, and financial signals), detects anomalies, clusters
events, and produces **analyst‑style intelligence summaries** using a
LangGraph agent workflow.

Agents operate **only on processed intelligence data**.\
They do **not perform raw ingestion, ETL, or heavy analytics**.

------------------------------------------------------------------------

# System Architecture

The system separates **data processing** from **agent reasoning**.

    DATA INGESTION
          ↓
    PROCESSING & DETECTION
          ↓
    INTELLIGENCE DATA LAYER
          ↓
    API SERVICE LAYER
          ↓
    AGENT REASONING LAYER
          ↓
    USER INTERFACES

This separation ensures:

-   deterministic analytics
-   fast response times
-   predictable agent behavior
-   easier debugging

Agents work exclusively on **structured intelligence outputs** rather
than raw datasets.

------------------------------------------------------------------------

# Ingestion Layer

The ingestion layer collects OSINT signals from external sources.

Primary ingestion scripts:

    telethon_update.py
    usgs_update.py
    news_update.py
    stocks_update.py

Typical cadence:

  Source           Frequency
  ---------------- ------------
  Telegram OSINT   15 minutes
  News feeds       hourly
  Earthquakes      15 minutes
  Market data      daily

These pipelines populate the **primary SQLite databases** used by the
system.

------------------------------------------------------------------------

# Processing Layer

After ingestion, scheduled jobs generate derived intelligence views.

These scripts create the **analytics layer used by agents**.

Examples:

    build_recent_clusters.py
    build_event_hourly_aggregates.py
    build_hourly_anomalies.py
    build_stock_alerts.py

Outputs include:

-   event burst clusters
-   hourly activity baselines
-   anomaly detection signals
-   financial alerts

These views allow the system to answer questions **without recomputing
heavy analytics**.

------------------------------------------------------------------------

# Intelligence Data Tables

Agents consume data from structured intelligence tables.

  Table                             Purpose
  --------------------------------- --------------------------------
  telegram_cluster_metrics_recent   recent burst activity clusters
  telegram_event_hourly             hourly activity baseline
  telegram_hourly_anomalies         anomaly detection signals
  stock_alerts                      unusual market activity
  earthquakes                       USGS seismic events

These tables represent the **interpretable intelligence layer** of the
system.

------------------------------------------------------------------------

# Design Principles

## Deterministic Processing First

All ingestion, normalization, clustering, and anomaly detection occurs
**outside the agent layer**.

Agents should never perform heavy analytics or large-scale
transformations.

------------------------------------------------------------------------

## Thin Retrieval, Strong Reasoning

Agents retrieve structured evidence quickly and focus on:

-   interpretation
-   prioritization
-   correlation
-   explanation

------------------------------------------------------------------------

## Cached Intelligence Preferred

Agents should use cached views whenever possible:

    /intel/latest
    /anomalies/hourly

This ensures fast responses and consistent outputs.

------------------------------------------------------------------------

## On‑Demand Analysis Only When Necessary

Custom computations should occur only for:

-   historical comparisons
-   custom time windows
-   research queries

------------------------------------------------------------------------

## Analyst‑Style Outputs

Responses should resemble **professional intelligence briefings**:

-   concise
-   evidence‑based
-   neutral tone
-   explicit uncertainty when needed

------------------------------------------------------------------------

# Shared Graph State

LangGraph agents operate on a shared state object.

Example structure:

``` python
class GraphState(TypedDict):
    user_request: str
    request_type: str
    time_window: str
    countries: list[str]
    event_types: list[str]
    sources_needed: list[str]
    retrieval_plan: dict
    retrieved_data: dict
    fused_analysis: dict
    forecast_analysis: dict
    final_report: str
```

------------------------------------------------------------------------

# Graph State Fields

  Field               Description
  ------------------- ---------------------------------------------
  user_request        original user query
  request_type        classified query type
  time_window         requested analysis window
  countries           geographic filters extracted from the query
  event_types         detected event categories
  sources_needed      intelligence datasets required
  retrieval_plan      structured instructions for retrieval
  retrieved_data      raw intelligence snapshot
  fused_analysis      interpreted intelligence findings
  forecast_analysis   near‑term implications
  final_report        formatted analyst report

------------------------------------------------------------------------

# Core Agents

The system currently uses five primary reasoning agents.

------------------------------------------------------------------------

# 1 Request Parser Agent

Purpose:

Convert the user's natural language request into structured intent.

Responsibilities:

-   classify request type
-   identify time window
-   detect countries
-   detect event types

Example:

Input:

    What happened near Iraq in the last hour?

Output:

``` json
{
  "request_type": "latest_summary",
  "time_window": "last_hour",
  "countries": ["Iraq"],
  "event_types": [],
  "sources_needed": ["telegram"]
}
```

------------------------------------------------------------------------

# 2 Query Planning Agent

Purpose:

Translate parsed intent into a **data retrieval strategy**.

Responsibilities:

-   determine which intelligence datasets to query
-   determine if cached results are sufficient
-   decide whether deeper analysis is required

Example output:

``` json
{
  "use_recent_clusters": true,
  "use_hourly_anomalies": true,
  "use_stock_alerts": false,
  "use_earthquakes": true,
  "requires_custom_analysis": false
}
```

------------------------------------------------------------------------

# 3 Retrieval Agent

Purpose:

Fetch structured intelligence evidence.

Typical sources:

    /intel/latest
    /anomalies/hourly
    cluster tables
    stock alerts
    earthquake feeds

Example result:

``` json
{
  "hourly_anomalies": [...],
  "recent_clusters": [...],
  "stock_alerts": [...],
  "earthquakes": [...]
}
```

Important rule:

Retrieval agents **do not interpret evidence**.

------------------------------------------------------------------------

# 4 Fusion Analyst Agent

Purpose:

Interpret signals across datasets and identify the most important
developments.

Responsibilities:

-   correlate signals
-   prioritize operational significance
-   assess confidence

Example output:

``` json
{
  "top_findings": [
    {
      "title": "Impact activity spike in Israel",
      "confidence": "high",
      "evidence": [
        "4x baseline anomaly",
        "multi-channel corroboration"
      ]
    }
  ],
  "overall_assessment": "Elevated operational tempo detected."
}
```

Evidence confidence categories may include:

-   confirmed event
-   strong signal
-   possible event
-   weak signal
-   uncertain

------------------------------------------------------------------------

# 5 Report Formatter Agent

Purpose:

Convert analysis into concise intelligence briefings.

Responsibilities:

-   produce readable summaries
-   highlight key developments
-   include evidence metrics
-   maintain neutral analytical tone

Example output:

    Operational activity increased during the last hour.

    Key developments:

    • Impact reports in Israel rose 3.8× above baseline.
    • Increased reporting activity near Erbil.
    • No significant earthquake activity detected.

    Overall assessment: moderate escalation in reported impact activity.

------------------------------------------------------------------------

# Routing Logic

The planner determines the correct analysis path.

Cached intelligence is used when:

-   the request concerns the last hour
-   the request concerns the last day
-   the user asks about anomalies
-   the user asks for summaries

Historical aggregates are used when:

-   the user asks about trends
-   comparisons are requested
-   multi‑day analysis is required

Custom recomputation is required when:

-   the requested window falls outside cached ranges
-   clustering must be rebuilt historically

------------------------------------------------------------------------

# Example Workflows

## Latest Intelligence Summary

User:

    What happened in the last hour?

Flow:

    Parser → Planner → Retrieval → Fusion → Formatter

------------------------------------------------------------------------

## Country‑Specific Activity

User:

    Show anomalies near Iraq

Flow:

    Parser → Planner → Retrieval → Fusion → Formatter

------------------------------------------------------------------------

## Historical Comparison

User:

    How does today compare to last week?

Flow:

    Parser → Planner → Retrieval → Fusion → Formatter

------------------------------------------------------------------------

# Prompting Guidelines

## Parser Agents

-   produce JSON outputs
-   avoid narrative responses
-   extract fields conservatively

## Fusion Agents

-   prioritize operationally significant signals
-   cite evidence clearly
-   acknowledge uncertainty

## Formatter Agents

-   concise
-   structured
-   neutral analytical tone

------------------------------------------------------------------------

# Failure Handling

Agents should gracefully handle incomplete intelligence.

Missing data:

    No significant anomalies detected.

Partial data:

Clearly indicate which sources were available.

Conflicting signals:

Explicitly state uncertainty.

------------------------------------------------------------------------

# Future Improvements

Planned enhancements include:

-   Telegram channel credibility scoring
-   entity‑aware geographic resolution
-   FIRMS thermal anomaly integration
-   long‑term historical clustering
-   automated daily intelligence briefings
-   dashboard‑based intelligence visualization

------------------------------------------------------------------------

# Summary

The agent layer converts structured intelligence data into **actionable
situational awareness**.

Primary goals:

-   detect anomalies
-   interpret signals
-   correlate sources
-   produce concise operational summaries
