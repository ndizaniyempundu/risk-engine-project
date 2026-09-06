# Concepts & Design Decisions

## Module 2 — Return Modeling
- **Log returns vs simple returns**: log returns are time-additive (sum of daily log returns = total log return over the period), which makes multi-day aggregation and statistics cleaner than simple percentage returns.
- **volatility() vs ewmaVol()**: plain volatility weights every historical day equally. EWMA (λ=0.94, RiskMetrics/JPM 1994) weights recent days more heavily, so it reacts faster to new market stress — this is why short-horizon risk desks prefer EWMA over a flat historical window.
- **Why annualize with √252**: variance scales linearly with time under a random-walk assumption, so standard deviation scales with the square root of time. 252 = approximate trading days per year.
- **Skewness & kurtosis prove non-normality empirically**: real asset returns show negative skew (crashes are sharper than rallies) and positive excess kurtosis (fat tails) — this is the direct motivation for why Module 3 uses Student-t copulas instead of assuming everything is Gaussian.
- **Jarque-Bera**: a single test statistic combining skew + kurtosis to formally test the "returns are normal" null hypothesis. Large JB = reject normality.

## Real result from SPY (2014–2024)
- Annualized volatility: ~17.2%
- Skewness: -0.80 (confirms real crashes are sharper than real rallies)
- Excess kurtosis: 13.99 (dramatically fatter tails than a normal distribution predicts — largely driven by the March 2020 COVID crash sitting in the dataset)
- Jarque-Bera: ~22,842 (normality is rejected at any reasonable significance threshold, which is typically around 6)