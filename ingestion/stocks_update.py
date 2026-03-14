import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
DATA_RAW_DIR = Path("db/")
DATA_PROCESSED_DIR = Path("db/")
DB_PATH = Path("db/market_data.db")

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = [
    "IBM",
    "META",
    "GOOGL",
    "LMT",   # Lockheed Martin
    "BA",    # Boeing
    "NOC",   # Northrop Grumman
    "GD",    # General Dynamics
    "RTX",   # RTX / Raytheon
    "LDOS",  # Leidos
    "LHX",   # L3Harris Technologies
    "AVAV",  # AeroVironment
    "NVDA",  # NVIDIA
    "INTC",  # Intel
    "PLTR",  # Palantir
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "CSCO",  # Cisco
    "CRM",   # Salesforce
    "AMZN",  # Amazon
    "MRK",   # Merck
    "JPM",   # JP Morgan
    "UNH",   # United Health
    "GS",    # Goldman Sachs
    "CAT",   # Caterpillar
    "HON",   # Honeywell
    "PG",    # Procter & Gamble
    "CVX",   # Chevron
    "WMT",   # Walmart
    "TSLA",  # Tesla
    "XOM",   # Exxon
    "JNJ",   # Johnson & Johnson
    "LIN",   # Linde
    "COST",  # Costco
    "MELI",  # Mercado Libre
    "CRWD",  # CrowdStrike
    "NU",    # Nu Holdings
    "GLD",   # Gold Shares ETF
    "KO",    # Coca-Cola
    "WM",    # Waste Management
    "ORCL",  # Oracle
    "MA",    # Mastercard
    "NFLX",  # Netflix
    "V",     # Visa
]

# Remove duplicates while preserving order
STOCKS = list(dict.fromkeys(STOCKS))


def download_stock_prices(stocks: list[str], period: str = "12mo") -> pd.DataFrame:
    """
    Download adjusted close/close prices from Yahoo Finance and return a wide dataframe:
    Date | AAPL | MSFT | ...
    """
    data = yf.download(
        tickers=stocks,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No stock data was returned by yfinance.")

    # Prefer Adjusted Close if available, otherwise Close
    if "Adj Close" in data.columns.get_level_values(0):
        price_df = data["Adj Close"].copy()
    elif "Close" in data.columns.get_level_values(0):
        price_df = data["Close"].copy()
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' was found in yfinance output.")

    price_df = price_df.reset_index()
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()

    return price_df


def make_long_format(price_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide stock table into long format:
    Date | Ticker | Price
    """
    long_df = price_wide.melt(
        id_vars=["Date"],
        var_name="Ticker",
        value_name="Price",
    )

    long_df["Date"] = pd.to_datetime(long_df["Date"]).dt.normalize()
    long_df["Ticker"] = long_df["Ticker"].astype(str)
    long_df["Price"] = pd.to_numeric(long_df["Price"], errors="coerce")

    long_df = long_df.dropna(subset=["Price"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return long_df


def build_stock_alerts(price_wide: pd.DataFrame, lookback_days: int = 3) -> pd.DataFrame:
    """
    Create event-style stock alerts from daily returns.
    Alerts:
      - Spike/Drop: absolute z-score > 3 on daily returns
      - 3-Day Decline: 3 consecutive negative-return days
    Returns only recent alerts in the last `lookback_days`.
    """
    returns = price_wide.set_index("Date").pct_change()
    alerts: list[dict] = []

    for ticker in returns.columns:
        series = pd.to_numeric(returns[ticker], errors="coerce").dropna()

        if series.empty:
            continue

        std = series.std()
        if pd.isna(std) or std == 0:
            z = pd.Series(index=series.index, dtype=float)
        else:
            z = (series - series.mean()) / std

        spike_dates = z[np.abs(z) > 3].index

        decline_mask = (series < 0).rolling(3).sum() >= 3
        decline_dates = decline_mask[decline_mask].index

        for d in spike_dates:
            alerts.append(
                {
                    "Date": pd.to_datetime(d).normalize(),
                    "Ticker": ticker,
                    "Alert": "Spike/Drop",
                }
            )

        for d in decline_dates:
            alerts.append(
                {
                    "Date": pd.to_datetime(d).normalize(),
                    "Ticker": ticker,
                    "Alert": "3-Day Decline",
                }
            )

    alerts_df = pd.DataFrame(alerts)

    if alerts_df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Alert"])

    alerts_df["Date"] = pd.to_datetime(alerts_df["Date"]).dt.normalize()

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    alerts_df = alerts_df[alerts_df["Date"] >= cutoff].copy()

    alerts_df = alerts_df.drop_duplicates().sort_values(["Date", "Ticker", "Alert"]).reset_index(drop=True)
    return alerts_df


def write_outputs(price_wide: pd.DataFrame, price_long: pd.DataFrame, alerts_df: pd.DataFrame) -> None:
    """
    Write both Excel exports and SQLite tables.
    """
    wide_excel = DATA_RAW_DIR / "StockData.xlsx"
    long_excel = DATA_PROCESSED_DIR / "StockData_long.xlsx"
    alerts_excel = DATA_PROCESSED_DIR / "stockalerts.xlsx"

    price_wide.to_excel(wide_excel, index=False)
    price_long.to_excel(long_excel, index=False)
    alerts_df.to_excel(alerts_excel, index=False)

    with sqlite3.connect(DB_PATH) as conn:
        price_wide.to_sql("stocks_data_wide", conn, if_exists="replace", index=False)
        price_long.to_sql("stocks_data_long", conn, if_exists="replace", index=False)
        alerts_df.to_sql("stock_alerts", conn, if_exists="replace", index=False)

    print(f"Wrote wide stock data to: {wide_excel}")
    print(f"Wrote long stock data to: {long_excel}")
    print(f"Wrote stock alerts to: {alerts_excel}")
    print(f"Wrote SQLite database to: {DB_PATH}")


def main() -> None:
    price_wide = download_stock_prices(STOCKS, period="12mo")
    price_long = make_long_format(price_wide)
    alerts_df = build_stock_alerts(price_wide, lookback_days=3)
    write_outputs(price_wide, price_long, alerts_df)

    print("\nSample stock prices:")
    print(price_wide.head())

    print("\nSample long-format prices:")
    print(price_long.head())

    print("\nRecent alerts:")
    print(alerts_df.head(20))


if __name__ == "__main__":
    main()