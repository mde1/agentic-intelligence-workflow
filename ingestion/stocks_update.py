import pandas as pd
import pandasai as pai
import requests
import os
from dotenv import load_dotenv
from io import StringIO
import yfinance as yf
import numpy as np

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

stocks = [
    "IBM",
    "META",
    "GOOG",
    "LMT", # Lockheed Martin
    "BA", # Boeing
    "NOC", # Northrup Gruman
    "GD", # General Dynamics
    "RTX", # Raytheon
    "LDOS", # Leidos
    "LHX", # L3Harris Technologies
    "AVAV", # AeroVironment
    "NVDA", #NVIDIA
    "INTC", # Intel
    "PLTR", # Palantir
    "AAPL", # Apple
    "MSFT", # Microsoft
    "CSCO", # Cisco
    "CRM", # Salesforce
    "AMZN", # Amazon
    "MRK", # Merck
    "JPM", # JP Morgan
    "UNH", # United Health
    "GS", # Goldman Sachs
    "CAT", # Caterpillar
    "HON", # Honeywell
    "BA", # Boeing
    "PG", # Proctor and Gamble
    "CVX", # Chevron
    "WMT", # Walmart

]

data = yf.download(stocks, period="12mo")

df = data["Close"].reset_index()
#df.index = df.index.to_flat_index()

#df = df.rename(columns={"Date": "Date"})

#print(df.head())
#print(df.columns)

df.to_excel("data/raw/StockData.xlsx", index=False)

def stock_outliers(df):
    # get daily change
    returns = df.set_index("Date").pct_change()

    alerts = []

    for ticker in returns.columns:

        series = returns[ticker]

        z = (series - series.mean()) / series.std()

        spike_dates = series[np.abs(z) > 3].index
        decline_dates = (series < 0).rolling(3).sum() >= 3

        for d in spike_dates:
            alerts.append({"Date": d, "Ticker": ticker, "Alert": "Spike/Drop"})

        for d in decline_dates[decline_dates].index:
            alerts.append({"Date": d, "Ticker": ticker, "Alert": "3-Day Decline"})

        alerts_df = pd.DataFrame(alerts)

        three_days_ago = pd.Timestamp.now() - pd.Timedelta(days=3)
        alerts_df = alerts_df[alerts_df['Date'] >= three_days_ago]
        print(alerts_df)

    return alerts_df

alerts = stock_outliers(df)
print(alerts)

