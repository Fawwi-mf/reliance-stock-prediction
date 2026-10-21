
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


# ============================================================
# RELIANCE AI - FINAL SIMPLE 3-MODEL DASHBOARD
# ============================================================
#
# Uses the final optimized models already trained:
#   1. Linear Regression
#   2. Random Forest
#   3. LSTM
#
# The selected model features are loaded from the saved
# feature-list files, so the dashboard does not hard-code
# a different feature set from the training pipeline.
#
# Refresh -> downloads the latest available Reliance data
# -> prepares the latest features -> predicts next trading day.
# ============================================================


st.set_page_config(
    page_title="Reliance AI",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #090d12;
            color: #e8edf4;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .brand {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .subtitle {
            color: #8995a6;
            margin-top: 4px;
            margin-bottom: 28px;
        }

        .section-title {
            color: #9ca8b8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            margin: 28px 0 12px 0;
        }

        .signal-up {
            color: #45d19a;
            font-weight: 800;
            font-size: 1.4rem;
        }

        .signal-down {
            color: #ff6b7a;
            font-weight: 800;
            font-size: 1.4rem;
        }

        .signal-neutral {
            color: #e0a84f;
            font-weight: 800;
            font-size: 1.4rem;
        }

        .stButton > button {
            background: #151c26;
            border: 1px solid #303b49;
            color: #e8edf4;
            border-radius: 8px;
        }

        .stButton > button:hover {
            background: #1b2430;
            border-color: #4b5a6d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SETTINGS
# ============================================================

TICKER = "RELIANCE.NS"
DATA_PERIOD = "2y"
SEQUENCE_LENGTH = 30

MODEL_PATH = "reliance_optimized_linear_model.pkl"
LR_SCALER_PATH = "reliance_optimized_linear_scaler.pkl"
RF_PATH = "reliance_optimized_random_forest.pkl"

LSTM_PATH = "reliance_optimized_lstm.keras"
LSTM_X_SCALER_PATH = "reliance_optimized_lstm_x_scaler.pkl"
LSTM_Y_SCALER_PATH = "reliance_optimized_lstm_y_scaler.pkl"

LR_FEATURES_PATH = "reliance_optimized_lr_features.pkl"
RF_FEATURES_PATH = "reliance_optimized_rf_features.pkl"
LSTM_FEATURES_PATH = "reliance_optimized_lstm_features.pkl"

RESULTS_PATH = "optimized_final_model_comparison.csv"


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_PATH,
    LR_SCALER_PATH,
    RF_PATH,
    LSTM_PATH,
    LSTM_X_SCALER_PATH,
    LSTM_Y_SCALER_PATH,
    LR_FEATURES_PATH,
    RF_FEATURES_PATH,
    LSTM_FEATURES_PATH,
]

missing_files = [
    filename
    for filename in required_files
    if not os.path.exists(filename)
]

if missing_files:
    st.error("Some final model files are missing:")
    for filename in missing_files:
        st.write(f"- {filename}")

    st.info(
        "Run feature_selection_optimization.py once more "
        "from the project folder to recreate the missing files."
    )
    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    linear_model = joblib.load(MODEL_PATH)
    lr_scaler = joblib.load(LR_SCALER_PATH)

    random_forest = joblib.load(RF_PATH)

    lstm_model = load_model(LSTM_PATH)
    lstm_x_scaler = joblib.load(LSTM_X_SCALER_PATH)
    lstm_y_scaler = joblib.load(LSTM_Y_SCALER_PATH)

    lr_features = joblib.load(LR_FEATURES_PATH)
    rf_features = joblib.load(RF_FEATURES_PATH)
    lstm_features = joblib.load(LSTM_FEATURES_PATH)

    return (
        linear_model,
        lr_scaler,
        random_forest,
        lstm_model,
        lstm_x_scaler,
        lstm_y_scaler,
        lr_features,
        rf_features,
        lstm_features,
    )


(
    linear_model,
    lr_scaler,
    random_forest,
    lstm_model,
    lstm_x_scaler,
    lstm_y_scaler,
    lr_features,
    rf_features,
    lstm_features,
) = load_models()


# ============================================================
# DOWNLOAD LATEST DATA
# ============================================================

@st.cache_data(ttl=900)
def get_stock_data():

    data = yf.download(
        TICKER,
        period=DATA_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data["Date"] = pd.to_datetime(data["Date"])

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["Date"] + numeric_columns
    )

    data = data.sort_values("Date")
    data = data.drop_duplicates("Date")
    data = data.reset_index(drop=True)

    return data


data = get_stock_data()


if data.empty:
    st.error(
        "Unable to retrieve the latest Reliance market data."
    )
    st.stop()


if len(data) < SEQUENCE_LENGTH:
    st.error(
        "Not enough market data for the LSTM."
    )
    st.stop()


# ============================================================
# TECHNICAL INDICATORS FOR DISPLAY
# ============================================================

data["MA_20"] = (
    data["Close"]
    .rolling(20)
    .mean()
)

data["MA_50"] = (
    data["Close"]
    .rolling(50)
    .mean()
)

data["Daily_Return"] = (
    data["Close"].pct_change()
)

data["Volatility"] = (
    data["Daily_Return"]
    .rolling(20)
    .std()
    * np.sqrt(252)
    * 100
)

delta = data["Close"].diff()

gain = (
    delta.clip(lower=0)
    .rolling(14)
    .mean()
)

loss = (
    (-delta.clip(upper=0))
    .rolling(14)
    .mean()
)

rs = gain / loss

data["RSI_14"] = (
    100 - (100 / (1 + rs))
)


# ============================================================
# LATEST VALUES
# ============================================================

latest = data.iloc[-1]
previous = data.iloc[-2]

current_price = float(latest["Close"])
previous_close = float(previous["Close"])

daily_change = (
    (current_price - previous_close)
    / previous_close
    * 100
)


# ============================================================
# PREDICTIONS
# ============================================================

def predict_next_day():

    # ----------------------------
    # Linear Regression
    # ----------------------------

    lr_input = (
        data[lr_features]
        .iloc[-1]
        .values
        .reshape(1, -1)
    )

    lr_scaled = lr_scaler.transform(
        lr_input
    )

    lr_prediction = float(
        linear_model.predict(
            lr_scaled
        )[0]
    )


    # ----------------------------
    # Random Forest
    # ----------------------------

    rf_input = (
        data[rf_features]
        .iloc[-1]
        .values
        .reshape(1, -1)
    )

    rf_prediction = float(
        random_forest.predict(
            rf_input
        )[0]
    )


    # ----------------------------
    # LSTM
    # ----------------------------

    lstm_input = (
        data[lstm_features]
        .tail(SEQUENCE_LENGTH)
        .values
    )

    lstm_scaled = (
        lstm_x_scaler.transform(
            lstm_input
        )
    )

    lstm_sequence = lstm_scaled.reshape(
        1,
        SEQUENCE_LENGTH,
        len(lstm_features),
    )

    lstm_scaled_prediction = (
        lstm_model.predict(
            lstm_sequence,
            verbose=0,
        )
    )

    lstm_prediction = float(
        lstm_y_scaler.inverse_transform(
            lstm_scaled_prediction
        )[0][0]
    )


    # ----------------------------
    # Simple ensemble
    # ----------------------------

    ensemble = (
        lr_prediction
        + rf_prediction
        + lstm_prediction
    ) / 3


    movement = (
        ensemble - current_price
    )

    change = (
        movement
        / current_price
        * 100
    )


    if change > 0.20:
        signal = "UP"
    elif change < -0.20:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"


    return (
        lr_prediction,
        rf_prediction,
        lstm_prediction,
        ensemble,
        movement,
        change,
        signal,
    )


(
    lr_prediction,
    rf_prediction,
    lstm_prediction,
    ensemble_forecast,
    expected_movement,
    expected_change,
    signal,
) = predict_next_day()


# ============================================================
# HEADER
# ============================================================

header1, header2 = st.columns(
    [6, 1]
)

with header1:

    st.markdown(
        '<div class="brand">RELIANCE AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent market analytics · NSE · RELIANCE.NS'
        '</div>',
        unsafe_allow_html=True,
    )


with header2:

    if st.button(
        "↻ Refresh",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Market Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
        f"{daily_change:+.2f}% today",
    )

with c2:
    st.metric(
        "AI Forecast",
        f"₹{ensemble_forecast:,.2f}",
        f"{expected_change:+.2f}% expected",
    )

with c3:
    st.metric(
        "Expected Move",
        f"₹{expected_movement:+,.2f}",
        f"{expected_change:+.2f}%",
    )

with c4:
    st.metric(
        "Signal",
        signal,
        "Next trading day",
    )


# ============================================================
# PRICE ACTION
# ============================================================

st.markdown(
    '<div class="section-title">Price Action</div>',
    unsafe_allow_html=True,
)

chart = go.Figure()

chart.add_trace(
    go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Reliance",
        increasing_line_color="#45d19a",
        decreasing_line_color="#ff6b7a",
    )
)

chart.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["MA_20"],
        name="MA 20",
        line=dict(
            color="#e0a84f",
            width=1.5,
        ),
    )
)

chart.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["MA_50"],
        name="MA 50",
        line=dict(
            color="#7d96ff",
            width=1.5,
        ),
    )
)

chart.add_trace(
    go.Scatter(
        x=[data["Date"].iloc[-1]],
        y=[ensemble_forecast],
        mode="markers",
        name="AI Forecast",
        marker=dict(
            size=10,
            color="#45d19a",
            symbol="diamond",
        ),
    )
)

chart.update_layout(
    height=520,
    template="plotly_dark",
    paper_bgcolor="#090d12",
    plot_bgcolor="#090d12",
    xaxis_rangeslider_visible=False,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
)

st.plotly_chart(
    chart,
    use_container_width=True,
)


# ============================================================
# MARKET SNAPSHOT + AI SIGNAL
# ============================================================

left, right = st.columns(2)

with left:

    st.markdown(
        '<div class="section-title">Market Snapshot</div>',
        unsafe_allow_html=True,
    )

    snapshot = pd.DataFrame(
        {
            "": [
                "Open",
                "Day High",
                "Day Low",
                "Volume",
                "Data Through",
            ],
            "Value": [
                f"₹{float(latest['Open']):,.2f}",
                f"₹{float(latest['High']):,.2f}",
                f"₹{float(latest['Low']):,.2f}",
                f"{int(latest['Volume']):,}",
                latest["Date"].strftime("%d %b %Y"),
            ],
        }
    )

    st.dataframe(
        snapshot,
        hide_index=True,
        use_container_width=True,
    )


with right:

    st.markdown(
        '<div class="section-title">AI Signal</div>',
        unsafe_allow_html=True,
    )

    signal_class = {
        "UP": "signal-up",
        "DOWN": "signal-down",
        "NEUTRAL": "signal-neutral",
    }[signal]

    st.markdown(
        f'<div class="{signal_class}">● {signal}</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"**Ensemble Forecast:** ₹{ensemble_forecast:,.2f}"
    )

    st.write(
        f"**Expected Difference:** "
        f"₹{expected_movement:+,.2f}"
    )

    st.write(
        f"**Expected Change:** "
        f"{expected_change:+.2f}%"
    )


# ============================================================
# TECHNICAL INDICATORS + MODEL PERFORMANCE
# ============================================================

left, right = st.columns(2)

with left:

    st.markdown(
        '<div class="section-title">Technical Indicators</div>',
        unsafe_allow_html=True,
    )

    indicators = pd.DataFrame(
        {
            "": [
                "RSI (14)",
                "MA (20)",
                "MA (50)",
                "20D Annualized Volatility",
            ],
            "Value": [
                f"{float(data['RSI_14'].iloc[-1]):.2f}",
                f"₹{float(data['MA_20'].iloc[-1]):,.2f}",
                f"₹{float(data['MA_50'].iloc[-1]):,.2f}",
                f"{float(data['Volatility'].iloc[-1]):.2f}%",
            ],
        }
    )

    st.dataframe(
        indicators,
        hide_index=True,
        use_container_width=True,
    )


with right:

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists(RESULTS_PATH):

        results = pd.read_csv(
            RESULTS_PATH
        )

        display_columns = [
            "Model",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
        ]

        available = [
            c
            for c in display_columns
            if c in results.columns
        ]

        st.dataframe(
            results[available],
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.info(
            "Model comparison file not found."
        )


# ============================================================
# INDIVIDUAL MODEL FORECASTS
# ============================================================

st.markdown(
    '<div class="section-title">Next-Day Forecast by Model</div>',
    unsafe_allow_html=True,
)

forecast_table = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "LSTM",
            "Ensemble",
        ],
        "Forecast": [
            f"₹{lr_prediction:,.2f}",
            f"₹{rf_prediction:,.2f}",
            f"₹{lstm_prediction:,.2f}",
            f"₹{ensemble_forecast:,.2f}",
        ],
        "Change vs Current": [
            f"{(lr_prediction-current_price)/current_price*100:+.2f}%",
            f"{(rf_prediction-current_price)/current_price*100:+.2f}%",
            f"{(lstm_prediction-current_price)/current_price*100:+.2f}%",
            f"{expected_change:+.2f}%",
        ],
    }
)

st.dataframe(
    forecast_table,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# RECENT TRADING DATA
# ============================================================

st.markdown(
    '<div class="section-title">Recent Trading Data</div>',
    unsafe_allow_html=True,
)

recent = data[
    [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
].tail(10).copy()

recent["Date"] = recent["Date"].dt.strftime(
    "%d %b %Y"
)

for column in [
    "Open",
    "High",
    "Low",
    "Close",
]:
    recent[column] = recent[column].map(
        lambda x: f"₹{x:,.2f}"
    )

recent["Volume"] = recent["Volume"].map(
    lambda x: f"{int(x):,}"
)

st.dataframe(
    recent,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Reliance AI is an academic forecasting system. "
    "Predictions are estimates, not guaranteed future prices "
    "or financial advice."
)
