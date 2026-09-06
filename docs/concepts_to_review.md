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

## Module 4 — Monte Carlo Engine
- **Variance reduction implemented**: antithetic variates (pairing each random draw z with -z), which measurably improved VaR estimate stability with fewer simulations.
- **Quasi-Monte Carlo (Sobol) noted as a future extension**: true Sobol sequence generation requires precomputed direction numbers and careful bit-manipulation logic that wasn't worth the implementation risk given the project timeline. A Halton sequence (same "low-discrepancy sequence" family, much simpler to implement correctly) would be the natural first step if extending this further.

## Module 5 — VaR & CVaR
- **Method divergence is the actual finding, not noise**: Historical Simulation's 99% CVaR ($28,408) is ~35% higher than Parametric's ($21,611) on the same portfolio -- this is Module 2's proven fat-tail/excess-kurtosis finding showing up concretely in dollar terms. Parametric VaR structurally cannot see this since it assumes normality.
- **Component VaR sums exactly to total VaR** (Euler's theorem for homogeneous functions) -- verified numerically: components summed to $13,275.0, matching total parametric VaR exactly.
- **NG=F drives ~89% of portfolio risk despite 20% capital weight** -- its 3.84% daily volatility (vs ~1% for other assets) dominates the risk budget even at equal dollar allocation. This is the practical meaning of Marginal VaR: it tells you which position to trim first if you need to cut risk.