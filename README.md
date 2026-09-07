# Multi-Asset Portfolio Risk Engine

A production-style risk engine spanning C++, Python, and SQL — computing VaR, CVaR, and stress-tested P&L across a 5-asset portfolio (equities, fixed income, FX, and commodities) using historical, parametric, and Monte Carlo methods, with copula-based tail-dependence modeling and pybind11-bound Python bindings feeding an interactive Streamlit dashboard.

**Stack:** C++17 · Eigen · SQLite · Python · pybind11 · Streamlit · Plotly

## Key Findings

- **Empirically proved non-normality**: SPY's 10-year return series shows excess kurtosis of ~14.0 and a Jarque-Bera statistic of ~22,842 (rejection threshold is ~6) — real returns are dramatically fat-tailed, not Gaussian.
- **Parametric VaR underestimates tail risk**: 99% CVaR under a Historical Simulation ($28,408) runs ~35% higher than under a Normal/Parametric assumption ($21,611) on the same portfolio — a direct, quantified consequence of the non-normality above.
- **Student-t copulas capture tail co-movement that Gaussian copulas miss**: at identical 0.7 correlation, a Student-t copula (ν=4) shows ~20% higher joint-tail probability than a Gaussian copula — the same structural blind spot that contributed to the 2008 financial crisis.
- **Concentration risk despite equal dollar weighting**: NG=F (natural gas futures) drives ~89% of total portfolio VaR despite being only 20% of invested capital, due to its outsized volatility (3.8% daily vs ~1% for other assets).
- **Backtesting confirms strong overall calibration** (5.22% breach rate vs a 5% target across 2,511 trading days) **but flags a Basel Traffic-Light RED zone in the most recent 250 days** — a breach-date diagnostic traces this directly back to NG=F, which was the largest single-asset contributor in all 12 recent breaches.

## Architecture

```
Data Layer (SQLite)
  assets | prices | positions | risk_metrics | stress_scenarios
        ↑ Python ETL (yfinance)
C++ Risk Engine Core
  ReturnSeries → CopulaEngine → MonteCarloEngine → VaR/CVaR → StressTester → Backtester
        ↓ pybind11 bindings
Python Layer
  Streamlit Dashboard (Plotly) — portfolio composition, VaR comparison, correlation
  heatmap, component VaR, stress scenarios, P&L distribution
```

## Repository Structure

```
src/          C++ source (ReturnSeries, CopulaEngine, MonteCarloEngine,
              VaR/CVaR calculator, StressTester, Backtester, pybind11 bindings)
python/       ETL pipeline, database builder, Python validation scripts,
              Streamlit dashboard
data/         Raw price CSVs (SQLite database is gitignored — see Setup)
docs/         Design decisions and methodology notes
```

## Setup & Running Locally

**1. C++ toolchain (macOS via Homebrew):**
```
brew install cmake eigen boost
```

**2. Python environment:**
```
python3 -m venv venv
source venv/bin/activate
pip install yfinance pandas scipy pybind11 streamlit plotly
```

**3. Build the database:**
```
python python/etl.py
python python/build_db.py
```

**4. Compile the C++ bindings** (the committed `.so` file is built for this project's specific Python 3.9/macOS environment — if it doesn't import cleanly on your machine, recompile it):
```
clang++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup -fPIC \
  -I/usr/local/opt/eigen/include/eigen3 \
  $(python3 -m pybind11 --includes) \
  src/bindings.cpp -o python/risk_engine$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
  -lsqlite3
```

**5. Launch the dashboard:**
```
streamlit run python/dashboard.py
```

## Methodology Notes

Detailed reasoning behind specific design decisions (why Historical Simulation over Parametric, EWMA vs. plain volatility, the Halton-vs-Sobol tradeoff, the reverse stress test's scenario-scaling approach, etc.) is documented in [`docs/concepts_to_review.md`](docs/concepts_to_review.md).

## Known Limitations & Future Extensions

- Asset date alignment is approximate (EURUSD=X trades on some days equities don't) — a minor simplification, not a correctness issue for the log-return calculations.
- Quasi-Monte Carlo uses a conceptual placeholder rather than a true Sobol sequence implementation (noted in `docs/concepts_to_review.md`).
- Position sizing is currently equal-weighted ($200K per asset) as a placeholder — the `positions` table schema supports real position data if extended.