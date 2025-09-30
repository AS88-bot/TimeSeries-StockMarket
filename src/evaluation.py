"""
src/evaluation.py

Provides evaluation metrics and utilities for time-series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def evaluate_forecast(y_true, y_pred):
    """
    Evaluates forecast using MAE, RMSE, MAPE.
    Returns a dictionary of results.
    """
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    _mape = mape(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": _mape}


def compare_models(y_true, forecasts_dict):
    """
    Compare multiple models' forecasts.
    forecasts_dict = {"ARIMA": series, "LSTM": series, ...}
    Returns a DataFrame with evaluation metrics for each model.
    """
    results = {}
    for name, pred in forecasts_dict.items():
        if isinstance(pred, (pd.Series, pd.DataFrame)):
            pred_values = pred.values.flatten()
        else:
            pred_values = np.array(pred).flatten()

        metrics = evaluate_forecast(y_true, pred_values)
        results[name] = metrics

    return pd.DataFrame(results).T


if __name__ == "__main__":
    # simple test
    y_true = [100, 102, 105, 107]
    y_pred = [99, 103, 106, 108]
    print("Single Evaluation:", evaluate_forecast(y_true, y_pred))

    forecasts = {
        "ModelA": [99, 103, 106, 108],
        "ModelB": [100, 101, 104, 106]
    }
    print("\nComparison:\n", compare_models(y_true, forecasts))
