import yfinance as yf
import numpy as np
import os

os.makedirs("data", exist_ok=True)

print("Downloading SPY, QQQ, IWM historical data...")
data = yf.download(["SPY", "QQQ", "IWM"], start="2020-01-01", end="2026-09-21", progress=False)
print(f"Downloaded {len(data)} rows")
print(f"Columns: {list(data.columns)}")

data.to_csv("data/real_market_data.csv")
print("Saved to data/real_market_data.csv")
print(data.head())
