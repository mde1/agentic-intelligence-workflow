#Agentic OSINT Intelligence Monitor
A LangGraph-powered intelligence monitoring system that aggregates open-source signals and produces real-time situational summaries.

The system ingests multiple OSINT data streams (Telegram, news feeds, market signals, seismic events), detects anomalies, and produces analyst-style intelligence reports via a conversational interface.

Currently the system operates through a Slack bot (Socket Mode), but the architecture is designed to evolve into a full web-based intelligence dashboard.

#System Overview

This project experiments with agentic intelligence workflows applied to real-time OSINT monitoring.

The system performs four core tasks:
- Collect signals from multiple OSINT sources
- Detect anomalies or unusual activity
- Fuse evidence across sources
- Generate structured intelligence summaries
- The result is an automated analyst capable of answering questions like:

```
What happened in the last hour?
Are there unusual spikes in Israel today?
Summarize major developments since the last update
```
# Architecture
The system uses a LangGraph agent workflow to process requests and generate reports.
``` Slack
  ↓
Slackbot (Socket Mode)
  ↓
LangGraph Agent
  ↓
FastAPI Data Services
  ↓
SQLite Intelligence Databases ```

#Agent Workflow
The core agent pipeline follows a structured reasoning process.

1. Request Parser
Interprets the user's natural-language request.
Example inputs:


Extracted signals:
time window
countries mentioned
event types
analysis mode