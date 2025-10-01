"""
fetch_data.py

Download historical stock data and save as data/cleaned_data.csv
Requires: yfinance
"""

import os
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2023-12-31", out_path="data/cleaned_data.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(out_path)
    print(f"Saved dataset to {out_path}")
    return df

if __name__ == "__main__":
    fetch_stock_data()
