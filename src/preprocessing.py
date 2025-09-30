import pandas as pd

def preprocess_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[["Close"]]  # focus only on closing price
    df = df.dropna()
    return df

if __name__ == "__main__":
    raw = pd.read_csv("../data/raw_data.csv")
    clean = preprocess_data(raw)
    clean.to_csv("../data/cleaned_data.csv")
    print("✅ Data cleaned and saved to data/cleaned_data.csv")
