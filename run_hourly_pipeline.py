import sys
from pipeline_utils import run_pipeline


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