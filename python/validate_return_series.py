import sqlite3
import numpy as np
from scipy import stats

conn = sqlite3.connect("data/risk_engine.db")
cur = conn.cursor()
cur.execute("SELECT close_price FROM prices WHERE asset_id = 1 ORDER BY date ASC")
prices = np.array([row[0] for row in cur.fetchall()])
conn.close()

log_returns = np.diff(np.log(prices))

mean_return = np.mean(log_returns)
volatility = np.std(log_returns, ddof=1) * np.sqrt(252)

# Manually replicate the C++ recursion (not pandas' .ewm()) for a true
# apples-to-apples check -- different libraries default to different
# EWMA conventions, so a library call wouldn't actually validate anything.
lam = 0.94
variance = log_returns[0] ** 2
for i in range(1, len(log_returns)):
    variance = lam * variance + (1 - lam) * log_returns[i - 1] ** 2
ewma_vol = np.sqrt(variance) * np.sqrt(252)

skewness = stats.skew(log_returns)       # scipy defaults match the C++ formulas
kurtosis = stats.kurtosis(log_returns)   # already "excess" kurtosis by default

n = len(log_returns)
jb = (n / 6) * (skewness**2 + (kurtosis**2) / 4)

print(f"Loaded {len(prices)} prices")
print(f"Mean return: {mean_return}")
print(f"Volatility (annualized): {volatility}")
print(f"EWMA volatility (annualized): {ewma_vol}")
print(f"Skewness: {skewness}")
print(f"Excess kurtosis: {kurtosis}")
print(f"Jarque-Bera statistic: {jb}")