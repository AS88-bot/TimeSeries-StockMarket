import os
import pandas as pd
from src.models import arima_forecast, lstm_forecast
from fetch_data import fetch_stock_data

def main():
    csv_path = os.path.join("data", "cleaned_data.csv")
    
    # Ensure data exists
    if not os.path.exists(csv_path):
        print("Data file missing. Fetching now...")
        fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2023-12-31", out_path=csv_path)

    # Load dataset
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Ensure 'Close' column is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])


    # Test ARIMA
    print("Running ARIMA forecast …")
    arima_preds = arima_forecast(df["Close"], steps=30)
    print("ARIMA forecast done. Sample:")
    print(arima_preds.head())

    # Test LSTM
    print("\nRunning LSTM forecast …")
    lstm_preds, scaler, history, model = lstm_forecast(
        df[["Close"]],
        lookback=60,
        epochs=5,
        test_size=0.2
    )
    print("LSTM forecast done. Sample:")
    print(lstm_preds.head())

    # Save predictions
    out_dir = os.path.join("results", "test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    arima_preds.to_csv(os.path.join(out_dir, "arima_test_preds.csv"))
    lstm_preds.to_csv(os.path.join(out_dir, "lstm_test_preds.csv"))
    print(f"\nPredictions saved under {out_dir}")

if __name__ == "__main__":
    main()
