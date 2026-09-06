import streamlit as st
import risk_engine

st.title("Risk Engine — Connection Test")

engine = risk_engine.RiskEngine("data/risk_engine.db", 5, [0.2, 0.2, 0.2, 0.2, 0.2], 1000000.0)

st.write("Observations loaded:", engine.num_observations())
st.write("Historical VaR (95%):", f"${engine.historical_var(0.95):,.2f}")
st.write("Parametric VaR (95%):", f"${engine.parametric_var(0.95):,.2f}")