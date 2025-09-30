"""
src/visualization.py

Functions for visualizing stock data, trends, and forecasts.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_stock(df, title="Stock Prices"):
    """
    Plots the stock closing price.
    df: DataFrame with 'Close' column and Date index
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_forecast(train, test, forecast, model_name="Model"):
    """
    Plots train, test, and forecasted values.
    train: pd.Series
    test: pd.Series
    forecast: pd.Series (aligned with test index ideally)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train.values, label="Train", color="black")
    plt.plot(test.index, test.values, label="Test", color="blue")
    plt.plot(forecast.index, forecast.values, label=f"{model_name} Forecast", color="red")
    plt.title(f"{model_name} Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_model_comparisons(test, forecasts_dict):
    """
    Plot multiple model forecasts against the true test set.
    test: pd.Series
    forecasts_dict: {"ARIMA": series, "LSTM": series, ...}
    """
    plt.figure(figsize=(12, 7))
    plt.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

    for name, pred in forecasts_dict.items():
        plt.plot(pred.index, pred.values, label=name)

    plt.title("Model Forecast Comparisons")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # quick test with dummy data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    train = pd.Series(range(1, 6), index=dates[:5])
    test = pd.Series(range(6, 11), index=dates[5:])
    forecast = pd.Series([5.5, 6.5, 7.5, 8.5, 9.5], index=dates[5:])

    plot_forecast(train, test, forecast, "DummyModel")

