# Agentic OSINT Intelligence Monitor

A **LangGraph-powered intelligence monitoring system** that aggregates
open-source signals and produces real-time situational summaries.

The system ingests multiple OSINT data streams (Telegram, news feeds,
market signals, seismic events), detects anomalies, and produces
**analyst-style intelligence reports** via a conversational interface.

Currently the system operates through a **Slack bot (Socket Mode)**, but
the architecture is designed to evolve into a **full web-based
intelligence dashboard**.

------------------------------------------------------------------------

# System Overview

This project experiments with **agentic intelligence workflows** applied
to real-time OSINT monitoring.

The system performs four core tasks:

1.  **Collect signals** from multiple OSINT sources\
2.  **Detect anomalies or unusual activity**
3.  **Fuse evidence across sources**
4.  **Generate structured intelligence summaries**

Example queries:

    What happened in the last hour?
    Are there unusual spikes in Israel today?
    Summarize major developments since the last update

------------------------------------------------------------------------

# Architecture

    Slack
      ↓
    Slackbot (Socket Mode)
      ↓
    LangGraph Agent
      ↓
    FastAPI Data Services
      ↓
    SQLite Intelligence Databases

------------------------------------------------------------------------

# Agent Workflow

    parse_request
        ↓
    plan_queries
        ↓
    retrieve_intel
        ↓
    fuse_findings
        ↓
    forecast_implications
        ↓
    format_response

### Request Parser

Extracts intent from natural language queries including:

-   time window
-   geography
-   event types
-   analysis mode

### Query Planner

Determines the **minimal set of data sources** needed to answer the
request.

### Retrieval Layer

Pulls intelligence snapshots from the API service.

### Fusion Node

Analyzes evidence across datasets and produces structured findings.

### Forecast Node

Generates near‑term implications and monitoring indicators.

### Report Formatter

Creates an analyst‑style intelligence summary.

------------------------------------------------------------------------

# Data Sources

## Telegram OSINT

Primary operational signal source.

Key tables:

-   `telegram_messages`
-   `telegram_cluster_metrics_recent`
-   `telegram_event_hourly`
-   `telegram_hourly_anomalies`

Used for:

-   incident reports
-   burst detection
-   anomaly scoring

------------------------------------------------------------------------

## News Feeds

RSS feeds provide broader geopolitical context and confirmation signals.

------------------------------------------------------------------------

## Defense Analysis

The War Zone RSS feed provides military and defense reporting context.

------------------------------------------------------------------------

## Earthquake Monitoring

USGS seismic feed allows correlation between seismic and operational
events.

------------------------------------------------------------------------

## Market Signals

Stock alerts provide economic indicators related to geopolitical
activity.

------------------------------------------------------------------------

# Running the System

## 1 Update Data Sources

    telethon_update.py
    wsj_update.py
    warzone_update.py
    usgs_update.py
    stocks_update.py

Typical cadence:

  Source        Frequency
  ------------- -----------
  Telegram      15 min
  News feeds    hourly
  Earthquakes   15 min
  Market data   daily

------------------------------------------------------------------------

## 2 Start the API

    python main.py

Endpoints include:

    /intel/latest
    /anomalies/hourly

------------------------------------------------------------------------

## 3 Start Slack Bot

    python bot.py

Example Slack usage:

    @intelbot what are the latest developments

------------------------------------------------------------------------

# Telegram API Requirement

Users must provide their **own Telegram API credentials**.

Create a `.env` file:

    TELEGRAM_API_ID=
    TELEGRAM_API_HASH=

------------------------------------------------------------------------

# Environment Variables

    OPENAI_API_KEY=
    SLACK_BOT_TOKEN=
    SLACK_APP_TOKEN=
    SLACK_SIGNING_SECRET=

    TELEGRAM_API_ID=
    TELEGRAM_API_HASH=

------------------------------------------------------------------------

# Project Structure

    agents/
        graph.py
        prompts/

    api/
        routes/

    db/

    bot.py
    main.py

------------------------------------------------------------------------

# Current Status

This project is **actively under development**.

Capabilities currently include:

-   OSINT ingestion
-   anomaly detection
-   cross‑source fusion
-   conversational intelligence summaries

------------------------------------------------------------------------

# Roadmap

Planned future improvements:

### Intelligence Dashboard

Transition from Slack output to a full web intelligence dashboard.

### Visual Analytics

-   anomaly charts
-   event heatmaps
-   regional timelines

### Multi‑Source Correlation

-   entity extraction
-   geospatial clustering
-   confidence scoring

### Persistent Memory

Track previously reported alerts and analyst annotations.

------------------------------------------------------------------------

# Philosophy

This system explores **agentic intelligence architectures** where LLMs
operate as reasoning components inside structured analytical pipelines.

The objective is to augment analysts by:

-   monitoring large OSINT streams
-   detecting emerging signals
-   generating structured situational awareness reports

------------------------------------------------------------------------

# Disclaimer

This project analyzes **open‑source information only** and should be
treated as an experimental analytical tool.
