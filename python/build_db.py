import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = "data/risk_engine.db"

TICKERS = {
    "SPY": {"asset_class": "Equity", "name": "S&P 500 ETF", "currency": "USD"},
    "IEF": {"asset_class": "Fixed Income", "name": "10Y Treasury ETF", "currency": "USD"},
    "EURUSD=X": {"asset_class": "FX", "name": "USD/EUR", "currency": "USD"},
    "GC=F": {"asset_class": "Commodity", "name": "Gold Futures", "currency": "USD"},
    "NG=F": {"asset_class": "Energy Commodity", "name": "Natural Gas Futures", "currency": "USD"},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY,
    ticker TEXT UNIQUE,
    asset_class TEXT,
    name TEXT,
    currency TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    asset_id INTEGER,
    date TEXT,
    close_price REAL,
    log_return REAL,
    FOREIGN KEY(asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS positions (
    asset_id INTEGER,
    quantity REAL,
    entry_price REAL,
    notional REAL,
    currency TEXT
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    date TEXT,
    var_95 REAL,
    var_99 REAL,
    cvar_95 REAL,
    cvar_99 REAL,
    method TEXT
);

CREATE TABLE IF NOT EXISTS stress_scenarios (
    scenario_name TEXT,
    asset_id INTEGER,
    shocked_return REAL,
    pnl_impact REAL
);
"""

def build_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()

    for ticker, info in TICKERS.items():
        cur.execute(
            "INSERT OR IGNORE INTO assets (ticker, asset_class, name, currency) VALUES (?, ?, ?, ?)",
            (ticker, info["asset_class"], info["name"], info["currency"])
        )
    conn.commit()

    cur.execute("SELECT id, ticker FROM assets")
    asset_ids = {ticker: asset_id for asset_id, ticker in cur.fetchall()}

    for ticker in TICKERS:
        safe_name = ticker.replace("=", "_")
        csv_path = Path(f"data/{safe_name}.csv")
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        asset_id = asset_ids[ticker]
        cur.execute("DELETE FROM prices WHERE asset_id = ?", (asset_id,))

        rows = [
            (asset_id, row.Date.strftime("%Y-%m-%d"), row.Close,
             None if pd.isna(row.log_return) else row.log_return)
            for row in df.itertuples()
        ]
        cur.executemany(
            "INSERT INTO prices (asset_id, date, close_price, log_return) VALUES (?, ?, ?, ?)",
            rows
        )
        conn.commit()
        print(f"  Loaded {len(rows)} rows for {ticker}")

    conn.close()
    print(f"Database built at {DB_PATH}")

if __name__ == "__main__":
    build_database()