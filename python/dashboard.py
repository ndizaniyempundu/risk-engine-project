import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import risk_engine

st.set_page_config(page_title="Multi-Asset Risk Engine", layout="wide")
st.title("Multi-Asset Portfolio Risk Engine")

ASSET_NAMES = ["SPY", "IEF", "EURUSD=X", "GC=F", "NG=F"]
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
PORTFOLIO_VALUE = 1_000_000.0

# @st.cache_resource keeps this from re-loading 2,763 rows of data and
# recomputing every statistic on every single UI interaction (like moving a
# slider) -- without it, the whole engine would rebuild on every click.
@st.cache_resource
def load_engine():
    return risk_engine.RiskEngine("data/risk_engine.db", 5, WEIGHTS, PORTFOLIO_VALUE)

engine = load_engine()

# --- Portfolio composition panel ---
st.header("Portfolio Composition")
comp_df = pd.DataFrame({
    "Asset": ASSET_NAMES,
    "Weight": WEIGHTS,
    "Notional ($)": [w * PORTFOLIO_VALUE for w in WEIGHTS]
})
col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(comp_df, hide_index=True)
with col2:
    fig_alloc = px.bar(comp_df, x="Asset", y="Weight", title="Allocation",
                        text_auto=".0%")
    fig_alloc.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_alloc, use_container_width=True)

# --- VaR comparison across methods ---
st.header("VaR Comparison: Historical vs Parametric vs Monte Carlo")
confidence_level = st.select_slider("Confidence Level", options=[0.95, 0.99], value=0.95)

var_data = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "VaR": [
        engine.historical_var(confidence_level),
        engine.parametric_var(confidence_level),
        engine.monte_carlo_var(confidence_level, 100000)
    ],
    "CVaR": [
        engine.historical_cvar(confidence_level),
        engine.parametric_cvar(confidence_level),
        engine.monte_carlo_cvar(confidence_level, 100000)
    ]
})
fig_var = go.Figure()
fig_var.add_trace(go.Bar(name="VaR", x=var_data["Method"], y=var_data["VaR"]))
fig_var.add_trace(go.Bar(name="CVaR", x=var_data["Method"], y=var_data["CVaR"]))
fig_var.update_layout(barmode="group", title=f"Risk at {int(confidence_level*100)}% Confidence")
st.plotly_chart(fig_var, use_container_width=True)
st.dataframe(var_data.style.format({"VaR": "${:,.2f}", "CVaR": "${:,.2f}"}), hide_index=True)

# --- Correlation heatmap ---
st.header("Asset Correlation Heatmap")
corr_matrix = engine.get_correlation_matrix()
corr_df = pd.DataFrame(corr_matrix, index=ASSET_NAMES, columns=ASSET_NAMES)
fig_heatmap = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="252-day Correlation Matrix")
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Marginal/Component VaR breakdown ---
st.header("Component VaR Breakdown")
component_var = engine.component_var(confidence_level)
comp_var_df = pd.DataFrame({"Asset": ASSET_NAMES, "Component VaR": component_var})
comp_var_df = comp_var_df.sort_values("Component VaR", ascending=False)
fig_comp = px.bar(comp_var_df, x="Asset", y="Component VaR",
                   title="Risk Contribution by Asset", text_auto=".2s")
st.plotly_chart(fig_comp, use_container_width=True)
st.caption(f"Sum of components: ${component_var.sum():,.2f} (should match total Parametric VaR)")

# --- Stress test ---
st.header("Stress Test Scenarios")
scenario_results = engine.run_all_scenarios()
scenario_df = pd.DataFrame(scenario_results, columns=["Scenario", "P&L"])
scenario_df = scenario_df.sort_values("P&L")

colors = ["crimson" if v < 0 else "seagreen" for v in scenario_df["P&L"]]
fig_stress = go.Figure(go.Bar(
    x=scenario_df["Scenario"],
    y=scenario_df["P&L"],
    text=[f"${v:,.0f}" for v in scenario_df["P&L"]],
    textposition="outside",
    marker_color=colors,
    cliponaxis=False
))
fig_stress.update_layout(
    title="Portfolio P&L by Stress Scenario",
    showlegend=False,
    yaxis_range=[scenario_df["P&L"].min() * 1.15, scenario_df["P&L"].max() * 1.15],
    margin=dict(t=60)
)
st.plotly_chart(fig_stress, use_container_width=True)

worst_idx = scenario_df["P&L"].idxmin()
st.caption(f"Worst-case scenario: {scenario_df.loc[worst_idx, 'Scenario']} "
           f"(Loss = ${abs(scenario_df.loc[worst_idx, 'P&L']):,.2f})")

# --- P&L distribution histogram with VaR/CVaR lines ---
st.header("Simulated Portfolio P&L Distribution")
pnl_dist = engine.get_portfolio_pnl_distribution(100000)
pnl_df = pd.DataFrame({"P&L": pnl_dist})

var_line = -engine.monte_carlo_var(confidence_level, 100000)
cvar_line = -engine.monte_carlo_cvar(confidence_level, 100000)

fig_hist = px.histogram(pnl_df, x="P&L", nbins=100, title="1-Day Simulated P&L Distribution (Monte Carlo)")

fig_hist.add_vline(x=var_line, line_dash="dash", line_color="orange",
                    annotation_text=f"VaR ({int(confidence_level*100)}%)",
                    annotation_position="top")
fig_hist.add_vline(x=cvar_line, line_dash="dash", line_color="red",
                    annotation_text=f"CVaR ({int(confidence_level*100)}%)",
                    annotation_position="bottom")
st.plotly_chart(fig_hist, use_container_width=True)