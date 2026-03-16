import sqlite3
from pathlib import Path

import pandas as pd
import requests


USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/1.0_day.geojson"
RAW_DIR = Path("data/raw")
DB_DIR = Path("db")
DB_PATH = DB_DIR / "market_osint.db"
TABLE_NAME = "usgs_earthquakes"


def fetch_usgs_earthquakes() -> pd.DataFrame:
    response = requests.get(USGS_URL, timeout=30)
    response.raise_for_status()
    data = response.json()

    rows = []

    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        coords = feature.get("geometry", {}).get("coordinates", []) or []

        longitude = coords[0] if len(coords) > 0 else None
        latitude = coords[1] if len(coords) > 1 else None
        depth_km = coords[2] if len(coords) > 2 else None

        row = {
            "usgs_id": feature.get("id"),
            "longitude": longitude,
            "latitude": latitude,
            "depth_km": depth_km,
            "magnitude": props.get("mag"),
            "title": props.get("title"),
            "location": props.get("place"),
            "type": props.get("type"),
            "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
            "updated_time": pd.to_datetime(props.get("updated"), unit="ms", errors="coerce"),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df["updated_time"] = pd.to_datetime(df["updated_time"])

    return df


def write_outputs(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    excel_path = RAW_DIR / "USGS.xlsx"
    df.to_excel(excel_path, index=False)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    print(len(df), "rows in the dataframe")
    print(f"Wrote Excel file to: {excel_path}")
    print(f"Wrote SQLite table '{TABLE_NAME}' to: {DB_PATH}")


def main() -> None:
    df = fetch_usgs_earthquakes()
    write_outputs(df)

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()