
# Agents Architecture

## Overview
This document defines the agent architecture for the **OSINT Anomaly Intelligence Platform**.

The system ingests data from Telegram channels, USGS earthquake feeds, news sources, and financial markets, then performs anomaly detection and event clustering. Agents operate on top of this processed intelligence layer to answer questions, generate summaries, and provide situational awareness.

Agents **do not perform raw ingestion or heavy ETL**. They reason over structured intelligence outputs.

---

# System Architecture

DATA INGESTION
↓
PROCESSING & DETECTION
↓
INTELLIGENCE DATA LAYER
↓
API ENDPOINTS
↓
AGENT REASONING LAYER
↓
USER INTERFACES

---

# Ingestion Layer

Scripts responsible for collecting data:

- telethon_update.py
- usgs_update.py
- news_update.py
- stocks_update.py

These scripts run on scheduled intervals and populate the primary database.

---

# Processing Layer

Scheduled scripts build derived intelligence tables:

- build_recent_clusters.py
- build_event_hourly_aggregates.py
- build_hourly_anomalies.py
- build_stock_alerts.py

Outputs include:

- burst activity clusters
- hourly activity baselines
- anomaly detection signals
- stock alerts

---

# Intelligence Data Tables

Primary tables consumed by agents:

| Table | Purpose |
|------|------|
| telegram_cluster_metrics_recent | recent event clusters |
| telegram_event_hourly | hourly activity baseline |
| telegram_hourly_anomalies | anomaly detection results |
| stock_alerts | market anomaly alerts |
| earthquakes | USGS earthquake data |

---

# Design Principles

## Deterministic processing first
All ingestion, normalization, clustering, and anomaly detection happens **outside the agent layer**.

## Thin retrieval, strong reasoning
Agents focus on interpretation and prioritization rather than data processing.

## Cached intelligence preferred
Agents should rely on cached intelligence views such as:

- /intel/latest
- /anomalies/hourly

## On-demand analysis when required
Custom computation should only occur for:

- historical queries
- custom time windows
- research analysis

## Analyst-style outputs
Agent responses should resemble concise intelligence briefings.

---

# Shared Graph State

Example LangGraph state structure:

```python
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
    final_response: str
```

### Field Descriptions

| Field | Description |
|------|-------------|
| user_request | original user question |
| request_type | classified query intent |
| time_window | requested analysis window |
| countries | extracted geographic filters |
| event_types | detected event categories |
| sources_needed | intelligence sources required |
| retrieval_plan | instructions for data retrieval |
| retrieved_data | structured evidence returned |
| fused_analysis | interpreted findings |
| final_response | formatted output |

---

# Core Agents

## 1. Request Parser Agent

Purpose:
Interpret the user's request and extract structured intent.

Responsibilities:

- classify request type
- identify time window
- detect geography
- detect event types

Example Input:

"What happened near Iraq in the last hour?"

Example Output:

```json
{
  "request_type": "latest_summary",
  "time_window": "last_hour",
  "countries": ["Iraq"],
  "event_types": [],
  "sources_needed": ["telegram"]
}
```

---

## 2. Query Planning Agent

Purpose:
Convert parsed intent into a retrieval strategy.

Responsibilities:

- determine which endpoints to query
- decide whether cached results are sufficient
- decide if deeper analysis is required

Example Output:

```json
{
  "use_recent_clusters": true,
  "use_hourly_anomalies": true,
  "use_stock_alerts": false,
  "use_earthquakes": true,
  "requires_custom_analysis": false
}
```

---

## 3. Retrieval Agent

Purpose:
Fetch structured intelligence data.

Sources:

- /intel/latest
- /anomalies/hourly
- cluster tables
- stock alerts
- earthquake data

Example Output:

```json
{
  "hourly_anomalies": [...],
  "recent_clusters": [...],
  "stock_alerts": [...],
  "earthquakes": [...]
}
```

Important rule:

Retrieval agents **do not interpret data**.

---

## 4. Fusion Analyst Agent

Purpose:
Interpret signals and identify meaningful developments.

Responsibilities:

- identify top developments
- correlate signals across sources
- evaluate severity and confidence

Example Output:

```json
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

Evidence confidence categories:

- confirmed event
- strong signal
- possible event
- weak signal
- uncertain

---

## 5. Report Formatter Agent

Purpose:
Convert fused analysis into final analyst-style summaries.

Responsibilities:

- produce concise outputs
- prioritize significant developments
- include key metrics
- maintain neutral tone

Example Output:

Operational activity increased during the last hour.

Key developments:

• Impact reports in Israel rose 3.8× above baseline.
• Increased reporting activity near Erbil.
• No significant earthquake activity detected.

Overall assessment: moderate escalation in reported impact activity.

---

# Routing Logic

Use cached intelligence when:

- request concerns the last hour
- request concerns the last day
- user asks for anomalies or summaries

Use historical aggregates when:

- user asks for trends
- user asks for comparisons
- multi-day analysis is required

Use custom recomputation when:

- requested window falls outside cached ranges
- clustering must be rebuilt historically

---

# Example Workflows

## Latest Summary

User:
"What happened in the last hour?"

Flow:

Request Parser → Query Planner → Retrieval Agent → Fusion Analyst → Formatter

---

## Country Specific Check

User:
"Show anomalies near Iraq"

Flow:

Parser → Planner → Retrieval → Fusion → Formatter

---

## Historical Comparison

User:
"How does today compare to last week?"

Flow:

Parser → Planner → Retrieve hourly metrics → Fusion → Formatter

---

# Prompting Guidelines

Parser Agents

- produce JSON outputs
- avoid narrative responses
- extract fields conservatively

Fusion Agents

- prioritize major developments
- cite evidence
- acknowledge uncertainty

Formatter Agents

- concise
- structured
- analyst tone

---

# Failure Handling

Missing data:

Return clear messages such as:

"No significant anomalies detected."

Partial data:

Indicate which sources were available.

Conflicting signals:

Explicitly note uncertainty.

---

# Future Improvements

Potential upgrades:

- Telegram channel credibility scoring
- entity-aware geographic resolution
- FIRMS thermal anomaly integration
- long-term historical clustering
- automated daily intelligence briefs

---

# Summary

The agent layer transforms structured intelligence data into **actionable insights**.

Goals:

- detect anomalies
- interpret signals
- correlate sources
- produce concise operational summaries
