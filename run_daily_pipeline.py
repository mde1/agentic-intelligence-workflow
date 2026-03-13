import sys
from pipeline_utils import run_pipeline

from pathlib import Path
from datetime import datetime, timezone

def write_success_marker(name: str):
    marker = Path("logs") / f"{name}_last_success.txt"
    marker.write_text(datetime.now(timezone.utc).isoformat())
    
def main() -> int:
    steps = [
        ("Stocks Update", "ingestion/stocks_update.py"),
        #("Build Stock Alerts", "processing/build_stock_alerts.py"),
    ]

    return run_pipeline("Daily Pipeline", steps)


if __name__ == "__main__":
    sys.exit(main())