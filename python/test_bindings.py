import risk_engine

engine = risk_engine.RiskEngine("data/risk_engine.db", 5, [0.2, 0.2, 0.2, 0.2, 0.2], 1000000.0)

print("Observations:", engine.num_observations())
print("Historical VaR 95%:", engine.historical_var(0.95))
print("Historical CVaR 95%:", engine.historical_cvar(0.95))
print("Parametric VaR 95%:", engine.parametric_var(0.95))
print("Monte Carlo VaR 95%:", engine.monte_carlo_var(0.95, 100000))
print("Component VaR:", engine.component_var(0.95))
print("Correlation matrix:\n", engine.get_correlation_matrix())