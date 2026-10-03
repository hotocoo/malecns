"""Download historical Forex data from Yahoo Finance.

Instruments: EURUSD=X, GBPUSD=X, USDJPY=X
Period: 5 years of daily data
Includes rate-limiting and retry logic for reliability.
"""

import csv
import os
import time
from datetime import datetime, timedelta

import requests
import yfinance as yf


def download_with_retry(ticker: str, period: str = "5y", interval: str = "1d",
                        max_retries: int = 3, base_delay: float = 2.0) -> "pd.DataFrame":
    """Download data with exponential backoff retry on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is not None and not df.empty:
                return df
            print(f"    Empty response for {ticker}")
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    Retry {attempt + 1}/{max_retries} for {ticker} in {delay:.0f}s: {e}")
                time.sleep(delay)
            else:
                print(f"    Failed after {max_retries} attempts for {ticker}: {e}")
                raise
    return None


def download_forex_data(output_file: str = "data/forex_market_data.csv", period: str = "5y"):
    """Download historical Forex data for major pairs."""
    instruments = {
        "EUR_USD": "EURUSD=X",
        "GBP_USD": "GBPUSD=X",
        "USD_JPY": "USDJPY=X",
    }

    print(f"Downloading Forex data for {list(instruments.keys())}...")
    print(f"Period: {period}")

    data = {}
    for i, (name, ticker) in enumerate(instruments.items()):
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = download_with_retry(ticker, period=period, interval="1d")
            if df is not None and not df.empty:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, tuple) or (hasattr(df.columns, 'names') and df.columns.names[0] is not None):
                    close_col = ('Close', ticker) if ('Close', ticker) in df.columns else 'Close'
                    high_col = ('High', ticker) if ('High', ticker) in df.columns else 'High'
                    low_col = ('Low', ticker) if ('Low', ticker) in df.columns else 'Low'
                    vol_col = ('Volume', ticker) if ('Volume', ticker) in df.columns else 'Volume'
                else:
                    close_col = 'Close'
                    high_col = 'High'
                    low_col = 'Low'
                    vol_col = 'Volume'
                data[name] = {
                    "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                    "close": df[close_col].values.astype(float),
                    "high": df[high_col].values.astype(float),
                    "low": df[low_col].values.astype(float),
                    "volume": df[vol_col].values.astype(float),
                }
                print(f"    Got {len(df)} days of data")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    Error: {e}")

        # Rate limiting: pause between downloads to avoid 429 errors
        if i < len(instruments) - 1:
            time.sleep(1.0)

    if not data:
        print("No data downloaded")
        return False

    # Find common dates
    all_dates = set(data[list(data.keys())[0]]["dates"])
    for name in data:
        all_dates &= set(data[name]["dates"])
    common_dates = sorted(all_dates)

    print(f"\nCommon trading days: {len(common_dates)}")

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["Date"]
        for name in data:
            header.extend([f"{name}_Close", f"{name}_High", f"{name}_Low", f"{name}_Volume"])
        writer.writerow(header)

        # Data
        for date in common_dates:
            row = [date]
            for name in data:
                idx = data[name]["dates"].index(date)
                row.extend([
                    f"{data[name]['close'][idx]:.5f}",
                    f"{data[name]['high'][idx]:.5f}",
                    f"{data[name]['low'][idx]:.5f}",
                    f"{int(data[name]['volume'][idx])}",
                ])
            writer.writerow(row)

    print(f"Data saved to {output_file}")
    return True


if __name__ == "__main__":
    download_forex_data()
