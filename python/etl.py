import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = {
    "SPY": "Equity",
    "IEF": "Fixed Income",
    "EURUSD=X": "FX",
    "GC=F": "Commodity",
    "NG=F": "Energy Commodity",
}

START_DATE = "2014-01-01"
END_DATE = "2024-12-31"

def fetch_data():
    all_data = {}
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"  WARNING: no data returned for {ticker}")
            continue
        print(f"  Got {len(df)} rows")
        all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    data = fetch_data()
    Path("data").mkdir(exist_ok=True)
    for ticker, df in data.items():
        safe_name = ticker.replace("=", "_")
        df.to_csv(f"data/{safe_name}.csv")
    print("Done. Raw CSVs saved in data/")

    