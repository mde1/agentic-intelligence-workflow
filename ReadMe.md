# Overall Workflow

## Run Main Update Scripts
1. telethon_update.py
2. wsj_update.py
3. warzone_update.py
4. usgs_update.py
5. stocks_update.py

## Run anomaly detection scoring
1. Latest events and severity

## Agent 1
1. Glean war locations (countries)
2. Run firms_update.py for countries where events occurred since last update
(maybe store each country's points in a separate dataframe or group by region)

## Agent 2
1. Analyze how the events might be interrelated

## Agent 3
1. Analyze the output and format a report

Node 1: request parser

Takes user request like:
“What happened in the last hour?”
“Show anomalies near Iraq”
“Summarize major changes since last update”

Outputs:
time window
geography
sources needed
task type

Node 2: planner
Determines what data to pull.
Example:
if geography requested → query Telegram + FIRMS + USGS + news
if market question → query stocks + related geopolitical events
if daily summary → pull latest fused alerts
This node should produce a structured plan, not prose.

Node 3: retrieval node
Runs the actual data queries.
This should call helper functions like:
get_recent_telegram_events()
get_recent_firms_by_country()
get_recent_earthquakes()
get_stock_anomalies()
get_fused_alerts()

Node 4: fusion/reasoning node
This is where the LLM actually earns its keep.
It looks across retrieved evidence and decides things like:
likely same event
weak correlation
no corroboration
possible rumor
likely escalation
likely unrelated
This is your real “intel” node.

Node 5: report formatter
Turns the fused output into:
Slack summary
analyst report
dashboard card text
alert blurb

Optional Node 6: memory/state node
Stores:
last update time
last countries checked
previous report context
recent alert IDs already reported
Useful, but keep it light.