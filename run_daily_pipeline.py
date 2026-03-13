import sys
from pipeline_utils import run_pipeline


def main() -> int:
    steps = [
        ("Stocks Update", "ingestion/stocks_update.py"),
        #("Build Stock Alerts", "processing/build_stock_alerts.py"),
    ]

    return run_pipeline("Daily Pipeline", steps)


if __name__ == "__main__":
    sys.exit(main())