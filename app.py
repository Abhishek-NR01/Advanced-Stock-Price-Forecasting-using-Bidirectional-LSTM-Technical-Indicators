import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import os


# Page config & basic styling
st.set_page_config(
    page_title="Quant Stock Forecast – BiLSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Single clean CSS block
st.markdown(
    """
    <style>
    /* Keep header visible for sidebar toggle */
    header {visibility: visible !important;}
    /* Hide Streamlit default menu & footer */
    #MainMenu, footer {visibility: hidden;}
    /* Metric cards styling */
    .metric-container {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, #020617, #111827);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #6b7280;
    }
    .block-container {
        padding-top: 3.0rem;   
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load Model, Scalers, Meta
MODEL_DIR = "."
model_path = os.path.join(MODEL_DIR, "bilstm_multistep.h5")
scaler_X_path = os.path.join(MODEL_DIR, "scaler_X.pkl")
scaler_y_path = os.path.join(MODEL_DIR, "scaler_y.pkl")
meta_path = os.path.join(MODEL_DIR, "meta.npy")

# Load scalers & meta info
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)
meta = np.load(meta_path, allow_pickle=True).item()

# Flatten feature_cols (handle possible MultiIndex)
raw_feature_cols = meta["feature_cols"]
feature_cols = []
for c in raw_feature_cols:
    if isinstance(c, tuple) and len(c) > 0:
        feature_cols.append(c[0])   # e.g. ('Open','AAPL') -> 'Open'
    else:
        feature_cols.append(c)

target_col = meta.get("target_col", "Close")
LSTM_WINDOW = meta["LSTM_WINDOW"]
FORECAST_HORIZON = meta["FORECAST_HORIZON"]
n_features = len(feature_cols)

# Rebuild BiLSTM architecture exactly as training time
def build_bilstm_model(lstm_window: int, n_features: int, horizon: int):
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(lstm_window, n_features)
        ),
        Bidirectional(
            LSTM(64)
        ),
        Dense(64, activation="relu"),
        Dense(horizon)
    ])
    return model

model = build_bilstm_model(LSTM_WINDOW, n_features, FORECAST_HORIZON)
# Treat .h5 as weights container: avoids Keras config deserialization issues
model.load_weights(model_path)


# Helper: Add technical indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)

    # Simple Moving Averages
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    # Bollinger Bands (20, 2)
    roll_mean = close.rolling(window=20).mean()
    roll_std = close.rolling(window=20).std()
    df["BB_High"] = roll_mean + 2 * roll_std
    df["BB_Low"] = roll_mean - 2 * roll_std

    # Daily Returns
    df["Returns"] = close.pct_change()

    # Remove rows with NaNs from indicators
    df = df.dropna()

    return df


# Sidebar – Controls
with st.sidebar:
    st.markdown("### ⚙️ Model Controls")
    ticker = st.text_input(
        "Symbol",
        value="AAPL",
        help="Any Yahoo Finance ticker (AAPL, TSLA, RELIANCE.NS, BTC-USD, etc.)",
    )
    period_years = st.slider("Lookback window (years)", 1, 10, 3)
    st.markdown("---")
    st.markdown(
        f"**Model:** Bidirectional LSTM  \n"
        f"**Input window:** {LSTM_WINDOW} days  \n"
        f"**Forecast horizon:** {FORECAST_HORIZON} business days"
    )
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)


# Main layout – Header
title_col, info_col = st.columns([3, 2])

with title_col:
    st.markdown(
        """
        <h2 style="margin-bottom:0.25rem;">Advanced Stock Forecast Desk</h2>
        <p style="color:#6b7280; margin-top:0;">
        Multivariate Bidirectional LSTM • Multi-step horizon • Indicator-aware forecast
        </p>
        """,
        unsafe_allow_html=True,
    )

with info_col:
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#9ca3af; text-align:right;">
        Educational only • Not trading or investment advice • Use at your own risk
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Placeholder for metrics
metrics_container = st.container()
st.markdown("")


# Core logic
if run_btn:
    with st.spinner("Pulling market data and running BiLSTM forecast…"):
        df = yf.download(ticker, period=f"{period_years}y")

        if df.empty:
            st.error("No data found for this ticker. Try a different symbol.")
        else:
            df_feat = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df_feat = add_indicators(df_feat)

            if len(df_feat) < (LSTM_WINDOW + 5):
                st.error(
                    "Not enough history after indicator calculation to build an input window. "
                    "Increase lookback period from the sidebar."
                )
            else:
                # Recent slice for model input
                df_recent = df_feat.iloc[-LSTM_WINDOW:]
                # Ensure order matches training
                df_recent = df_recent[feature_cols]

                X_input = df_recent.values  # (window, n_features)
                X_scaled = scaler_X.transform(X_input)
                X_scaled = X_scaled.reshape(1, LSTM_WINDOW, len(feature_cols))

                # Predict
                y_pred_scaled = model.predict(X_scaled)
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()

                # Future dates (business days)
                last_date = df_feat.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=FORECAST_HORIZON,
                    freq="B",
                )

                forecast_df = pd.DataFrame(
                    {"Date": future_dates, "Predicted_Close": y_pred_inv}
                ).set_index("Date")

                # ====== Metrics (top row) ======
                last_close = float(df_feat["Close"].iloc[-1])
                next_1d = float(y_pred_inv[0])
                last_horizon = float(y_pred_inv[-1])

                pct_1d = (next_1d - last_close) / last_close * 100
                pct_horizon = (last_horizon - last_close) / last_close * 100

                col_m1, col_m2, col_m3 = metrics_container.columns(3)

                with col_m1:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-label">Last Close ({ticker.upper()})</div>
                            <div class="metric-value">${last_close:,.2f}</div>
                            <div class="metric-sub">Spot reference</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_m2:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-label">Next Session Forecast</div>
                            <div class="metric-value">${next_1d:,.2f}</div>
                            <div class="metric-sub">{pct_1d:+.2f}% vs last close</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_m3:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-label">{FORECAST_HORIZON}-Day Horizon</div>
                            <div class="metric-value">${last_horizon:,.2f}</div>
                            <div class="metric-sub">{pct_horizon:+.2f}% over horizon</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("")

                # ====== Tabs: Charts & Table ======
                tab1, tab2, tab3 = st.tabs(
                    ["📉 Forecast vs Recent", "🕯️ Candles + Forecast", "📊 Forecast Table"]
                )

                # Tab 1 – Matplotlib line plot
                with tab1:
                    st.markdown("#### Price trajectory with forecast overlay")
                    fig, ax = plt.subplots(figsize=(11, 4))
                    recent_window = df_feat.index[-100:]
                    ax.plot(
                        recent_window,
                        df_feat["Close"].iloc[-100:],
                        label="Recent Actual Close",
                        linewidth=1.4,
                    )
                    ax.plot(
                        forecast_df.index,
                        forecast_df["Predicted_Close"],
                        marker="o",
                        linestyle="--",
                        label=f"BiLSTM Forecast ({FORECAST_HORIZON}d)",
                        linewidth=1.4,
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(alpha=0.2)
                    st.pyplot(fig, clear_figure=True)

                # Tab 2 – Plotly candlestick + forecast
                with tab2:
                    st.markdown("#### Candlestick view with projected path")

                    recent_ohlc = df_feat.iloc[-100:].copy()
                    fig_c = go.Figure(
                        data=[
                            go.Candlestick(
                                x=recent_ohlc.index,
                                open=recent_ohlc["Open"],
                                high=recent_ohlc["High"],
                                low=recent_ohlc["Low"],
                                close=recent_ohlc["Close"],
                                name="Recent OHLC",
                            )
                        ]
                    )
                    fig_c.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df["Predicted_Close"],
                            mode="lines+markers",
                            name="Forecast Close",
                        )
                    )
                    fig_c.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=450,
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_c, use_container_width=True)

                # Tab 3 – Table
                with tab3:
                    st.markdown("#### Forecasted levels (business days)")
                    st.dataframe(
                        forecast_df.style.format({"Predicted_Close": "{:,.2f}"}),
                        use_container_width=True,
                    )

else:
    st.info(
        "Configure the symbol and lookback window from the sidebar, then hit **Run Forecast** "
        "to generate a BiLSTM-based multi-step forecast."
    )