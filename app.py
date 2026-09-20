"""
app.py
======
Streamlit dashboard: Predictive Forecasting of Care Load & Placement Demand.

Run with:
    streamlit run app.py

Expects a CSV at data/raw_uac_data.csv with columns:
    Date, Children apprehended and placed in CBP custody,
    Children in CBP custody, Children transferred out of CBP custody,
    Children in HHS Care, Children discharged from HHS Care

If that file isn't found, the app falls back to a synthetic demo dataset
so the dashboard is fully explorable out of the box.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.evaluate import compute_all_kpis, evaluate_forecast, horizon_error
from src.models import (
    ARIMAForecaster,
    ExpSmoothingForecaster,
    MLForecaster,
    MovingAverage,
    NaivePersistence,
    SARIMAForecaster,
)
from src.preprocessing import generate_synthetic_uac_data, prepare_dataset

st.set_page_config(
    page_title="UAC Care Load & Placement Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

DATA_PATH = "data/raw_uac_data.csv"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = prepare_dataset(DATA_PATH)
        source = "Uploaded dataset"
    else:
        df = generate_synthetic_uac_data()
        source = "Synthetic demo dataset (place your CSV at data/raw_uac_data.csv to use real data)"
    return df, source


df, data_source = load_data()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Forecast Controls")
st.sidebar.caption(data_source)

target_options = {
    "Children in HHS Care": "hhs_care_load",
    "Children discharged from HHS Care": "hhs_discharges",
}
target_label = st.sidebar.selectbox("Series to forecast", list(target_options.keys()))
target_col = target_options[target_label]

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=60, value=14, step=1)

model_choices = st.sidebar.multiselect(
    "Models to compare",
    ["Naive Persistence", "Moving Average", "ARIMA", "SARIMA", "Exp. Smoothing",
     "Random Forest", "Gradient Boosting"],
    default=["SARIMA", "Random Forest"],
)

capacity_threshold = st.sidebar.number_input(
    "Capacity threshold (for breach probability & surge lead time)",
    min_value=0.0,
    value=float(df[target_col].quantile(0.9)),
    step=50.0,
)

holdout_days = st.sidebar.slider(
    "Backtest holdout window (days, for accuracy metrics)", 7, 60, 14
)

st.sidebar.divider()
scenario_shock = st.sidebar.slider(
    "Scenario: intake surge (%)", min_value=-30, max_value=100, value=0, step=5,
    help="Simulate a sudden change in CBP transfer volume to stress-test the forecast."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Predictive Forecasting of Care Load & Placement Demand")
st.caption("U.S. Department of Health and Human Services — UAC Program")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest HHS Care Load", f"{int(df['hhs_care_load'].iloc[-1]):,}")
col2.metric("Latest Daily Discharges", f"{int(df['hhs_discharges'].iloc[-1]):,}")
col3.metric("7-day Avg Net Pressure",
            f"{(df['cbp_transfers_out'] - df['hhs_discharges']).tail(7).mean():+.0f}")
col4.metric("Data Range", f"{df.index.min().date()} → {df.index.max().date()}")

st.divider()

# ---------------------------------------------------------------------------
# Train/backtest split
# ---------------------------------------------------------------------------
train = df.iloc[:-holdout_days]
test = df.iloc[-holdout_days:]
y_train = train[target_col]
y_test = test[target_col]

results = {}
forecasts = {}
ci_bounds = {}

MODEL_REGISTRY = {
    "Naive Persistence": lambda: NaivePersistence(),
    "Moving Average": lambda: MovingAverage(window=7),
    "ARIMA": lambda: ARIMAForecaster(order=(2, 1, 2)),
    "SARIMA": lambda: SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)),
    "Exp. Smoothing": lambda: ExpSmoothingForecaster(seasonal_periods=7),
    "Random Forest": lambda: MLForecaster(model_type="random_forest", target_col=target_col),
    "Gradient Boosting": lambda: MLForecaster(model_type="gradient_boosting", target_col=target_col),
}

with st.spinner("Fitting models..."):
    for name in model_choices:
        try:
            model = MODEL_REGISTRY[name]()
            if name in ("Random Forest", "Gradient Boosting"):
                model.fit(train)
                preds = model.forecast(holdout_days)
                lower = upper = None
            elif name in ("ARIMA", "SARIMA", "Exp. Smoothing"):
                model.fit(y_train)
                preds, lower, upper = model.forecast(holdout_days)
            else:
                model.fit(y_train)
                preds = model.forecast(holdout_days)
                lower = upper = None

            results[name] = evaluate_forecast(y_test.values, preds)
            forecasts[name] = preds
            ci_bounds[name] = (lower, upper)
        except Exception as e:
            st.warning(f"{name} failed to fit: {e}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Forecast Chart", "🔀 Model Comparison", "🎯 Confidence Intervals", "🚨 KPIs & Capacity"]
)

# --- Tab 1: Forecast chart ---
with tab1:
    st.subheader(f"Backtest: {target_label}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index[-60:], y=y_train.iloc[-60:], name="Train (recent)", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=test.index, y=y_test, name="Actual", line=dict(color="black", width=3)))
    colors = ["#2E86AB", "#E63946", "#2A9D8F", "#F4A261", "#8338EC", "#FB5607", "#3A86FF"]
    for i, (name, preds) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(
            x=test.index, y=preds, name=name, line=dict(color=colors[i % len(colors)], dash="dash")
        ))
    fig.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forward-looking forecast")
    best_model_name = min(results, key=lambda k: results[k]["RMSE"]) if results else None
    if best_model_name:
        st.info(f"Best backtest model by RMSE: **{best_model_name}**")
        full_model = MODEL_REGISTRY[best_model_name]()
        if best_model_name in ("Random Forest", "Gradient Boosting"):
            full_model.fit(df)
            future_preds = full_model.forecast(horizon)
            future_lower = future_upper = None
        elif best_model_name in ("ARIMA", "SARIMA", "Exp. Smoothing"):
            full_model.fit(df[target_col])
            future_preds, future_lower, future_upper = full_model.forecast(horizon)
        else:
            full_model.fit(df[target_col])
            future_preds = full_model.forecast(horizon)
            future_lower = future_upper = None

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)

        # scenario adjustment: shock applied proportionally to forecast
        shocked_preds = future_preds * (1 + scenario_shock / 100.0) if scenario_shock else future_preds

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[-60:], y=df[target_col].iloc[-60:], name="History", line=dict(color="gray")))
        fig2.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Forecast", line=dict(color="#2E86AB")))
        if scenario_shock:
            fig2.add_trace(go.Scatter(x=future_dates, y=shocked_preds, name=f"Scenario ({scenario_shock:+d}%)", line=dict(color="#E63946", dash="dot")))
        if future_lower is not None:
            fig2.add_trace(go.Scatter(x=future_dates, y=future_upper, line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=future_dates, y=future_lower, line=dict(width=0), fill="tonexty",
                                       fillcolor="rgba(46,134,171,0.15)", name="95% CI"))
        fig2.add_hline(y=capacity_threshold, line_dash="dash", line_color="red",
                        annotation_text="Capacity threshold")
        fig2.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Model comparison ---
with tab2:
    st.subheader("Model Accuracy Comparison (holdout backtest)")
    if results:
        comp_df = pd.DataFrame(results).T.sort_values("RMSE")
        st.dataframe(comp_df.style.background_gradient(cmap="RdYlGn_r", subset=["MAE", "RMSE", "MAPE (%)"]),
                     use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=comp_df.index, y=comp_df["RMSE"], name="RMSE", marker_color="#2E86AB"))
        fig3.update_layout(height=350, title="RMSE by Model (lower is better)")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Per-horizon error breakdown (best model)")
        if best_model_name:
            herr = horizon_error(y_test.values, forecasts[best_model_name])
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=herr["horizon_day"], y=herr["pct_error"], marker_color="#F4A261"))
            fig4.update_layout(height=300, xaxis_title="Days ahead", yaxis_title="% Error",
                                title=f"{best_model_name}: error growth by forecast horizon")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Select at least one model in the sidebar to compare.")

# --- Tab 3: Confidence intervals ---
with tab3:
    st.subheader("Forecast Uncertainty")
    ci_models = [m for m in ci_bounds if ci_bounds[m][0] is not None]
    if ci_models:
        chosen = st.selectbox("Model", ci_models)
        lower, upper = ci_bounds[chosen]
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=test.index, y=upper, line=dict(width=0), showlegend=False))
        fig5.add_trace(go.Scatter(x=test.index, y=lower, line=dict(width=0), fill="tonexty",
                                   fillcolor="rgba(46,134,171,0.2)", name="95% CI"))
        fig5.add_trace(go.Scatter(x=test.index, y=forecasts[chosen], name="Forecast", line=dict(color="#2E86AB")))
        fig5.add_trace(go.Scatter(x=test.index, y=y_test, name="Actual", line=dict(color="black")))
        fig5.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Confidence intervals are available for ARIMA, SARIMA, and Exponential Smoothing models.")

# --- Tab 4: KPIs ---
with tab4:
    st.subheader("Operational KPIs")
    if best_model_name:
        kpis = compute_all_kpis(y_test.values, forecasts[best_model_name], capacity_threshold, test.index)
        kcols = st.columns(len(kpis))
        for (k, v), c in zip(kpis.items(), kcols):
            c.metric(k, v if v is not None else "No breach forecast")

        st.divider()
        st.subheader("Capacity Breach Risk")
        if future_lower is not None and future_upper is not None:
            breach_days = future_dates[future_preds >= capacity_threshold]
            if len(breach_days) > 0:
                st.error(f"⚠️ Forecast crosses capacity threshold starting **{breach_days[0].date()}** "
                          f"({(breach_days[0] - df.index[-1]).days} days from now).")
            else:
                st.success("✅ No capacity breach forecast within the selected horizon.")
        st.caption("Net daily pressure (transfers in − discharges out), last 90 days")
        pressure = (df["cbp_transfers_out"] - df["hhs_discharges"]).tail(90)
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=pressure.index, y=pressure.values,
                               marker_color=np.where(pressure.values >= 0, "#E63946", "#2A9D8F")))
        fig6.update_layout(height=300)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Fit at least one model to see KPIs.")

st.divider()
st.caption(
    "Predictive Forecasting of Care Load & Placement Demand · "
    "UAC Program · Unified Mentor / U.S. Department of Health and Human Services"
)
