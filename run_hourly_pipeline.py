import sys
from pipeline_utils import run_pipeline
from pathlib import Path
from datetime import datetime, timezone

def write_success_marker(name: str):
    marker = Path("logs") / f"{name}_last_success.txt"
    marker.write_text(datetime.now(timezone.utc).isoformat())

def main() -> int:
    steps = [
        ("News Update", "ingestion/wsj_update.py"),
        ("Warzone Update", "ingestion/warzone_update.py"),
        # Add more hourly context steps here later if needed
        # ("Normalize News", "processing/build_news_events.py"),
    ]

    return run_pipeline("Hourly Pipeline", steps)


if __name__ == "__main__":
    sys.exit(main())