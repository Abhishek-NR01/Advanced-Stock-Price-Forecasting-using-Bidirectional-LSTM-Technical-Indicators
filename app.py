import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

import os

# Load Model & Scalers

MODEL_DIR = "."
model_path = os.path.join(MODEL_DIR, "bilstm_multistep.h5")
scaler_X_path = os.path.join(MODEL_DIR, "scaler_X.pkl")
scaler_y_path = os.path.join(MODEL_DIR, "scaler_y.pkl")
meta_path = os.path.join(MODEL_DIR, "meta.npy")

model = load_model(model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)
meta = np.load(meta_path, allow_pickle=True).item()

feature_cols = meta["feature_cols"]
target_col = meta["target_col"]
LSTM_WINDOW = meta["LSTM_WINDOW"]
FORECAST_HORIZON = meta["FORECAST_HORIZON"]

# Helper: add indicators

def add_indicators(df):
    df = df.copy()
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    return df

# Streamlit UI

st.title("📈 Advanced Stock Price Forecasting (BiLSTM, Multi-step)")

st.write("""
This app uses a multivariate, multi-step **Bidirectional LSTM** model trained on:
- OHLCV data
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Returns)  
to forecast future stock prices for the next **N days**.
""")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS):", "AAPL")
period_years = st.slider("How many past years to use for context?", 1, 10, 3)

if st.button("Run Forecast"):
    # 1. Download recent data
    df = yf.download(ticker, period=f"{period_years}y")
    if df.empty:
        st.error("No data found for this ticker.")
    else:
        st.subheader("Recent Close Price")
        st.line_chart(df['Close'])

        # 2. Add indicators
        df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df_feat = add_indicators(df_feat)

        if len(df_feat) < (LSTM_WINDOW + 5):
            st.error("Not enough data after indicators to form a window. Try a longer period.")
        else:
            # 3. Take last LSTM_WINDOW rows
            df_recent = df_feat.iloc[-LSTM_WINDOW:]
            st.write(f"Using last {LSTM_WINDOW} days of data to predict next {FORECAST_HORIZON} days.")

            # Ensure column ordering matches training
            # (if columns changed, you may need to manually align)
            df_recent = df_recent[feature_cols]

            X_input = df_recent.values  # (window, n_features)
            X_scaled = scaler_X.transform(X_input)
            X_scaled = X_scaled.reshape(1, LSTM_WINDOW, len(feature_cols))

            # 4. Predict
            y_pred_scaled = model.predict(X_scaled)  # (1, horizon)
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()

            # 5. Prepare future dates
            last_date = df_feat.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='B')  # business days

            # 6. Show results
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_Close": y_pred_inv
            }).set_index("Date")

            st.subheader("Forecasted Future Prices")
            st.dataframe(forecast_df)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_feat.index[-100:], df_feat['Close'].iloc[-100:], label="Recent Actual Close")
            ax.plot(forecast_df.index, forecast_df['Predicted_Close'], marker='o', label="Forecasted Close")
            ax.set_title(f"{ticker} - Next {FORECAST_HORIZON}-Day Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
