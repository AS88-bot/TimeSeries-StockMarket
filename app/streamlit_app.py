"""
app/streamlit_app.py

Interactive dashboard for stock price forecasting.
Uses ARIMA, SARIMA, Prophet, and LSTM models.
"""

import streamlit as st
import pandas as pd
import yfinance as yf

from src.preprocessing import preprocess_data
from src.models import compare_models_on_series
from src.visualisation import plot_model_comparisons
from src.evaluation import compare_models


# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="📊 Stock Market Forecasting", layout="wide")

st.title("📈 Stock Market Forecasting Dashboard")
st.markdown("Compare **ARIMA, SARIMA, Prophet, and LSTM** models for stock price forecasting.")


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, MSFT, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
test_size = st.sidebar.slider("Test Size (%)", 10, 30, 15) / 100

run_button = st.sidebar.button("Run Forecast")


# -----------------------------
# Fetch & preprocess data
# -----------------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

if run_button:
    with st.spinner("Fetching stock data..."):
        raw = load_data(ticker, start_date, end_date)
        df = preprocess_data(raw)

    st.subheader(f"📊 {ticker} Closing Price Data")
    st.line_chart(df["Close"])

    # -------------------------
    # Run Models
    # -------------------------
    st.subheader("🤖 Running Models... (this may take a moment)")

    metrics, forecasts = compare_models_on_series(
        df,
        test_size=test_size,
        arima_order=(5, 1, 0),
        sarima_order=(1, 1, 1),
        sarima_seasonal=(1, 1, 1, 5),
        lstm_lookback=60,
        lstm_epochs=10   # keep low for speed
    )

    # -------------------------
    # Show Results
    # -------------------------
    st.subheader("📊 Model Performance Metrics")
    st.write(pd.DataFrame(metrics).T)

    st.subheader("📈 Forecast Comparison")
    test = df[int(len(df)*(1-test_size)):]["Close"]
    plot_model_comparisons(test, forecasts)
