import sys
from pipeline_utils import run_pipeline

from pathlib import Path
from datetime import datetime, timezone

def write_success_marker(name: str):
    marker = Path("logs") / f"{name}_last_success.txt"
    marker.write_text(datetime.now(timezone.utc).isoformat())

def main() -> int:
    steps = [
        ("Telegram Update", "ingestion/telethon_update.py"),
        ("USGS Update", "ingestion/usgs_update.py"),
        ("Build Recent Clusters", "processing/build_recent_clusters.py"),
        ("Build Hourly Aggregates", "processing/build_event_hourly_aggregates.py"),
        ("Build Hourly Anomalies", "processing/build_hourly_anomalies.py"),
    ]

    return run_pipeline("Realtime Pipeline", steps)


if __name__ == "__main__":
    sys.exit(main())