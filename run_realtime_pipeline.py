import sys
from pipeline_utils import run_pipeline


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