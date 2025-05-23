import numpy as np
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- API key Alpha Vantage ka ---
API_KEY = "S40393YA8YDCJNMV"  # Yahan apna API key daalna

# --- Global Settings ---
INTERVAL = "60min"         # Time interval
WINDOW_SIZE = 60           # LSTM ko 60 points chahiye
EPOCHS = 5
BATCH_SIZE = 32

# --- Stock ka data API se fetch karna ---
def fetch_stock_data(symbol: str, interval: str) -> pd.DataFrame:
    print(f"[INFO] '{symbol}' ka data fetch ho raha hai...")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": "full"
    }

    response = requests.get(url, params=params)
    data = response.json()

    key = f"Time Series ({interval})"
    if key not in data:
        raise Exception(f"[ERROR] Stock symbol '{symbol}' sahi nahi hai ya API se data nahi aaya.")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }).astype(float)

    df = df.sort_index()  # Time order mein la rahe hain
    return df

# --- Data ko model ke liye prepare karna ---
def prepare_data(df: pd.DataFrame, feature: str = "Close"):
    print("[INFO] Data normalize ho raha hai...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature]])

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data)):
        X.append(scaled_data[i - WINDOW_SIZE:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# --- LSTM model ka structure ---
def build_lstm_model(input_shape):
    print("[INFO] Model ban raha hai...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# --- Prediction karna next price ke liye ---
def predict_next_close(model, df: pd.DataFrame, scaler: MinMaxScaler):
    last_sequence = df["Close"].values[-WINDOW_SIZE:]
    scaled_seq = scaler.transform(last_sequence.reshape(-1, 1))
    input_seq = scaled_seq.reshape((1, WINDOW_SIZE, 1))
    prediction = model.predict(input_seq)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

# --- Price chart dikhana ---
def plot_closing_prices(df: pd.DataFrame, symbol: str):
    print("[INFO] Price chart ban raha hai...")
    df["Close"].tail(100).plot(title=f"{symbol} ka Closing Price (last 100 points)", figsize=(10, 5))
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main function ---
def main():
    print(f"\n[{datetime.datetime.now()}] 📈 Stock Price Predictor Start ho gaya hai...\n")
    stock_symbol = input("🧾 Enter stock symbol (jaise AAPL, MSFT, TSLA): ").upper()

    try:
        df = fetch_stock_data(stock_symbol, INTERVAL)
    except Exception as e:
        print(e)
        return

    X, y, scaler = prepare_data(df)
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    predicted_price = predict_next_close(model, df, scaler)
    print(f"\n💸 {stock_symbol} ka next close price prediction: ${predicted_price:.2f}\n")

    plot_closing_prices(df, stock_symbol)

# --- Run karne wala block ---
if __name__ == "__main__":
    main()

