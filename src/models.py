"""
src/models.py

Implements time-series forecasting models:
 - ARIMA (statsmodels)
 - SARIMA (SARIMAX - statsmodels)
 - Prophet (facebook/prophet)
 - LSTM (tensorflow.keras)

Utilities: train/test split (time-series), evaluation metrics (MAE, RMSE, MAPE),
and a small example in __main__ showing how to run each model on
../data/cleaned_data.csv with a Date index and a 'Close' column.

Notes:
 - Prophet package is installed as `prophet` (modern versions). If you have
   fbprophet installed, change import accordingly.
 - LSTM uses MinMaxScaler and iterative multi-step forecasting.
 - Adjust hyperparameters for your dataset/compute.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ARIMA / SARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
try:
    from prophet import Prophet
except Exception:
    # fallback if older package name is installed
    from fbprophet import Prophet  # type: ignore

# LSTM
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM functionality will be disabled.")
    # Create dummy classes to prevent import errors
    class Sequential: pass
    class LSTM: pass
    class Dense: pass
    class Dropout: pass
    class EarlyStopping: pass
    tf = None

# reproducibility
SEED = 42
np.random.seed(SEED)
if TENSORFLOW_AVAILABLE and tf is not None:
    tf.random.set_seed(SEED)


# -------------------------
# Utility functions
# -------------------------
def train_test_split_time_series(series, test_size=0.2):
    """
    Splits a pandas Series or DataFrame (with datetime index) into train/test by time.
    Returns (train, test).
    """
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        n = len(series)
        split = int(np.ceil(n * (1 - test_size)))
        train = series.iloc[:split]
        test = series.iloc[split:]
        return train, test
    else:
        raise ValueError("Input must be a pandas Series or DataFrame.")


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (in percent). Avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def evaluate_forecast(y_true, y_pred):
    """
    Returns a dict with MAE, RMSE, MAPE.
    y_true and y_pred must be 1-D arrays (same length).
    """
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    _mape = mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": _mape}


# -------------------------
# ARIMA
# -------------------------
def arima_forecast(train_series, steps=30, order=(5, 1, 0), return_model=False):
    """
    Fit ARIMA on train_series (pd.Series) and forecast `steps` ahead.
    Returns forecast (pd.Series). If return_model=True returns (forecast, fitted_model).
    """
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    fc = model_fit.forecast(steps=steps)
    fc.index = pd.date_range(start=pd.Timestamp(train_series.index[-1]) + pd.Timedelta(1, unit='D'),
                             periods=steps, freq=pd.infer_freq(train_series.index) or 'D')
    if return_model:
        return fc, model_fit
    return fc


# -------------------------
# SARIMA (seasonal ARIMA)
# -------------------------
def sarima_forecast(train_series, steps=30, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5), return_model=False):
    """
    Fit SARIMAX (SARIMA) on train_series and forecast `steps` ahead.
    seasonal_order is (P, D, Q, s)
    """
    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    fc = model_fit.get_forecast(steps=steps)
    pred = fc.predicted_mean
    freq = pd.infer_freq(train_series.index) or 'D'
    pred.index = pd.date_range(start=pd.Timestamp(train_series.index[-1]) + pd.Timedelta(1, unit='D'),
                               periods=steps, freq=freq)
    if return_model:
        return pred, model_fit
    return pred


# -------------------------
# Prophet
# -------------------------
def prophet_forecast(train_df, periods=30, freq=None, seasonality_mode='additive', return_model=False):
    """
    train_df: pd.DataFrame with DatetimeIndex and a 'Close' column (or a df already with 'ds','y').
    periods: steps to forecast.
    freq: frequency string for future dataframe. If None, tries to infer.
    """
    # Prepare dataframe for Prophet
    if 'ds' in train_df.columns and 'y' in train_df.columns:
        dfp = train_df[['ds', 'y']].copy()
    else:
        dfp = train_df[['Close']].copy()
        dfp = dfp.reset_index().rename(columns={'Date': 'ds', dfp.columns[1]: 'y'}) if 'Date' in dfp.columns else dfp.reset_index().rename(columns={dfp.index.name or dfp.index.name: 'ds', 'Close': 'y'})
        # above fallback handles typical DataFrame; ensure columns are ds & y
        dfp.columns = ['ds', 'y']

    # infer freq if not provided
    if freq is None:
        try:
            freq = pd.infer_freq(pd.to_datetime(dfp['ds']))
            if freq is None:
                freq = 'D'
        except Exception:
            freq = 'D'

    model = Prophet(seasonality_mode=seasonality_mode, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(dfp)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    # extract only forecast rows beyond training
    pred = forecast[['ds', 'yhat']].set_index('ds').yhat[-periods:]
    if return_model:
        return pred, model
    return pred


# -------------------------
# LSTM (deep learning)
# -------------------------
def create_sequences(data, lookback=60):
    """
    Given a 2D array (n_samples, n_features=1), create sequences for LSTM.
    Returns X (samples, lookback, 1), y (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def lstm_forecast(series, lookback=60, epochs=20, batch_size=32, units=50, test_size=0.2, verbose=0):
    """
    Train an LSTM on the series (pd.Series or DataFrame with 'Close') and forecast the test set length.
    Returns: (predictions_series, scaler, history_dict, model)
      - predictions_series: pd.Series indexed by test dates
    Note: This function uses iterative forecasting (predict one step, append, predict next).
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install tensorflow to use LSTM functionality.")
    # Prepare values
    if isinstance(series, pd.DataFrame):
        values = series['Close'].values.reshape(-1, 1)
        index = series.index
    else:
        values = series.values.reshape(-1, 1)
        index = series.index

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # split
    split = int(np.ceil(len(scaled) * (1 - test_size)))
    train_scaled = scaled[:split]
    test_scaled = scaled[split:]

    # create sequences for training
    if len(train_scaled) <= lookback:
        raise ValueError("Train size must be larger than lookback. Reduce lookback or test_size.")
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)

    # build model
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(max(8, units // 2)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[es])

    # iterative multi-step forecast for len(test_scaled)
    preds = []
    last_window = train_scaled[-lookback:].reshape(lookback, 1)

    steps = len(test_scaled)
    for _ in range(steps):
        x_input = last_window.reshape((1, lookback, 1))
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)
        # roll window
        last_window = np.vstack([last_window[1:], [[yhat]]])

    # inverse transform
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).reshape(-1)

    # build pandas Series with same index as test
    test_index = index[split:]
    preds_series = pd.Series(data=preds_inv, index=test_index, name='LSTM_Pred')

    return preds_series, scaler, history.history, model


# -------------------------
# Example runner and model comparison
# -------------------------
def compare_models_on_series(df, test_size=0.2, arima_order=(5, 1, 0),
                             sarima_order=(1, 1, 1), sarima_seasonal=(1, 1, 1, 5),
                             prophet_periods=None,
                             lstm_lookback=60, lstm_epochs=20):
    """
    Given df with Date index and 'Close', split into train/test and run all models.
    Returns a dict of metrics and a dict of forecasts (each a pd.Series indexed by test dates).
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must have a 'Close' column.")

    train, test = train_test_split_time_series(df[['Close']], test_size=test_size)
    results = {}
    forecasts = {}

    # ARIMA
    arima_steps = len(test)
    try:
        arima_fc = arima_forecast(train['Close'], steps=arima_steps, order=arima_order)
        arima_fc = arima_fc[:arima_steps]
        arima_fc.index = test.index  # align indexes for evaluation
        forecasts['ARIMA'] = arima_fc
        results['ARIMA'] = evaluate_forecast(test['Close'].values, arima_fc.values)
    except Exception as e:
        results['ARIMA'] = {"error": str(e)}

    # SARIMA
    try:
        sarima_fc = sarima_forecast(train['Close'], steps=arima_steps, order=sarima_order, seasonal_order=sarima_seasonal)
        sarima_fc = sarima_fc[:arima_steps]
        sarima_fc.index = test.index
        forecasts['SARIMA'] = sarima_fc
        results['SARIMA'] = evaluate_forecast(test['Close'].values, sarima_fc.values)
    except Exception as e:
        results['SARIMA'] = {"error": str(e)}

    # Prophet
    try:
        freq = pd.infer_freq(train.index) or 'D'
        prophet_periods = prophet_periods or arima_steps
        p_fc = prophet_forecast(train, periods=prophet_periods, freq=freq)
        p_fc = p_fc[:arima_steps]
        p_fc.index = test.index
        forecasts['Prophet'] = p_fc
        results['Prophet'] = evaluate_forecast(test['Close'].values, p_fc.values)
    except Exception as e:
        results['Prophet'] = {"error": str(e)}

    # LSTM
    try:
        lstm_fc, _, _, _ = lstm_forecast(df[['Close']], lookback=lstm_lookback, epochs=lstm_epochs, test_size=test_size, verbose=0)
        # ensure index alignment and length (lstm returns Series indexed by test index)
        lstm_fc = lstm_fc[:len(test)]
        forecasts['LSTM'] = lstm_fc
        results['LSTM'] = evaluate_forecast(test['Close'].values, lstm_fc.values)
    except Exception as e:
        results['LSTM'] = {"error": str(e)}

    return results, forecasts


# -------------------------
# Command-line / script example
# -------------------------
if __name__ == "__main__":
    import os
    print("Running models.py example...")

    # path expected relative to src/
    path = os.path.join("data", "cleaned_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}. Please run the data_loader + preprocessing scripts first.")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    # ensure column named 'Close'
    if 'Close' not in df.columns:
        raise ValueError("cleaned_data.csv must contain a 'Close' column")

    # run comparison (this may take time because LSTM trains)
    metrics, forecasts = compare_models_on_series(df, test_size=0.15,
                                                 arima_order=(5, 1, 0),
                                                 sarima_order=(1, 1, 1),
                                                 sarima_seasonal=(1, 1, 1, 5),
                                                 lstm_lookback=60,
                                                 lstm_epochs=15)  # reduce epochs for example
    print("\n=== Metrics ===")
    for model_name, metric in metrics.items():
        print(f"{model_name}: {metric}")

    # optional: save forecasts to CSV
    out_dir = os.path.join("..", "results")
    os.makedirs(out_dir, exist_ok=True)
    for name, ser in forecasts.items():
        if isinstance(ser, pd.Series):
            ser.to_csv(os.path.join(out_dir, f"forecast_{name}.csv"))

    print(f"\nForecasts saved to {out_dir} (if generated). Done.")

