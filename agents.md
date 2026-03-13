# Agents Architecture

## Purpose
This document defines the agent architecture for the OSINT anomaly intelligence platform.
The system monitors Telegram, USGS earthquake data, news, and market alerts, then provides:
- analyst-style summaries
- anomaly explanations
- cross-source reasoning
- Slack-ready responses
- dashboard-facing intelligence outputs

The agent layer does not perform raw ingestion or heavy ETL. It sits on top of processed tables and API endpoints.
---

## Design Principles
1. **Deterministic data processing first**
   - Data fetching, normalization, clustering, and anomaly detection should happen in scheduled scripts.
   - Agents should reason over structured outputs, not raw chaos.

2. **Thin retrieval, strong reasoning**
   - Retrieval nodes gather the right processed data.
   - Reasoning nodes interpret significance, corroboration, uncertainty, and impact.

3. **Fast operational answers**
   - Common requests should use precomputed recent clusters and anomaly tables.

4. **On-demand deep analysis only when necessary**
   - Custom historical clustering or backfills should happen only for special queries.

5. **Analyst-style outputs**
   - Agents should produce concise, evidence-based summaries with clear uncertainty language.

---

## Shared State
The LangGraph state should remain compact and structured.

```python
class GraphState(TypedDict):
    user_request: str
    request_type: str
    time_window: str
    countries: list[str]
    event_types: list[str]
    sources_needed: list[str]
    retrieved_data: dict
    fused_analysis: dict
    final_response: str