"""Module for enhanced bot with CNN+LSTM+Attention, multi-step forecast, and news sentiment."""

import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import torch
import torch.nn as nn


@st.cache_resource(show_spinner=False)
def _ensure_vader():
    nltk.download("vader_lexicon", quiet=True)
    return True


_ = _ensure_vader()


ORIGINAL_CNN_LSTM_FEATURES = [
    "Close", "MA_Short", "MA_Long",
    "Lag_1", "Lag_2", "Lag_3",
    "Range", "Log_Return",
    "RSI", "MACD", "MACD_Signal", "MACD_Diff",
    "Bollinger_High", "Bollinger_Low",
    "Stoch_K", "Stoch_D"
]
ALL_FEATURES = ORIGINAL_CNN_LSTM_FEATURES + ["News_Sentiment"]


# session-state defaults
if "initial_balance" not in st.session_state:
    st.session_state.initial_balance = 20_000
if "balance" not in st.session_state:
    st.session_state.balance = st.session_state.initial_balance

# ------------------------------------------------------
# 3) Data Fetching
# ------------------------------------------------------


def fetch_live_data(tickers, retries=3):
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        st.error("API key not found for FinancialModelingPrep.")
        return data

    for ticker in tickers:
        for attempt in range(retries):
            try:
                ticker_api = ticker.replace('/', '')
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                if not data_json or len(data_json) < 1:
                    st.warning(f"No data returned for {ticker}.")
                    continue

                df = pd.DataFrame(data_json)
                if df.empty:
                    st.warning(f"No data available for {ticker}.")
                    continue

                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                }, inplace=True)
                df.sort_index(inplace=True)

                data[ticker] = df
                break  # success
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(
                        f"Retrying for {ticker}... Attempt {attempt + 1}")
                else:
                    st.error(f"Failed to fetch data for {ticker}: {e}")
        else:
            st.error(
                f"Failed to fetch data for {ticker} after {retries} attempts.")
    return data

# ------------------------------------------------------
# 4) News Sentiment Analysis
# ------------------------------------------------------


def fetch_news(ticker):
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    if not news_api_key:
        st.error("API key not found for NewsAPI.org.")
        return []

    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={news_api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        st.error("Failed to fetch news articles.")
        return []


def compute_sentiment_score(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        text = ' '.join(filter(None, [title, description, content]))
        if text:
            score = sia.polarity_scores(text)['compound']
            sentiment_scores.append(score)
    if sentiment_scores:
        average_score = np.mean(sentiment_scores)
    else:
        average_score = 0
    return average_score

# ------------------------------------------------------
# 5) Indicator Computations
# ------------------------------------------------------


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line


def compute_bollinger_bands(df, period=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + num_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - num_std * df['BB_Std']
    return df


def compute_stochastic_oscillator(df, k_period=14, d_period=3):
    min_low = df['Low'].rolling(window=k_period).min()
    max_high = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * (df['Close'] - min_low) / (max_high - min_low)
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df


def compute_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

# ------------------------------------------------------
# 6) Data Preparation
# ------------------------------------------------------


def prepare_data(df, sentiment_score):
    """
    Prepare the data by adding lag features, indicators, and handling NaN values.
    """
    # Add lag features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)

    # Add technical indicators
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    df = compute_bollinger_bands(df)
    df['Bollinger_High'] = df['BB_Upper']
    df['Bollinger_Low'] = df['BB_Lower']
    df = compute_stochastic_oscillator(df)
    df['Stoch_K'] = df['%K']
    df['Stoch_D'] = df['%D']
    df = compute_atr(df)

    # Add moving averages
    df['MA_Short'] = df['Close'].rolling(5).mean()
    df['MA_Long'] = df['Close'].rolling(20).mean()
    # Additional features
    df['Range'] = df['High'] - df['Low']
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Add news sentiment score
    df['News_Sentiment'] = sentiment_score

    # Drop rows with NaN values resulting from rolling or shifting
    df.dropna(inplace=True)

    return df

# ------------------------------------------------------
# 7) Utility / Predictions
# ------------------------------------------------------


def predict_price_change_percentage(current_price, predicted_price):
    if current_price == 0 or predicted_price is None:
        return 0
    return ((predicted_price - current_price) / current_price) * 100


def compute_indicators(df, sentiment_score):
    df = df.copy()
    if 'Close' not in df.columns:
        st.warning("Data does not contain 'Close' column.")
        return df

    # This function already computes all needed indicators
    df = prepare_data(df, sentiment_score)

    return df


def multi_step_forecast(model, X_last, n_steps, scaler):
    forecasts = []
    X_input = torch.tensor(X_last, dtype=torch.float32)
    for _ in range(n_steps):
        with torch.no_grad():
            pred = model(X_input).numpy()
        forecasts.append(pred[0])
        X_np = X_input.numpy()
        X_np = np.roll(X_np, -1, axis=1)
        X_np[0, -1, 0] = pred[0]  # update 'Close'
        X_input = torch.tensor(X_np, dtype=torch.float32)
    # Inverse transform forecasts
    inv_forecasts = []
    for pred in forecasts:
        dummy = np.zeros((1, len(ORIGINAL_CNN_LSTM_FEATURES)))
        dummy[0, 0] = pred
        inv_pred = scaler.inverse_transform(dummy)
        inv_close = inv_pred[0, 0]
        inv_forecasts.append(inv_close)
    return inv_forecasts

# ------------------------------------------------------
# 8) Plot Predictions
# ------------------------------------------------------


def plot_predictions(df, ensemble_pred_value_adjusted, forecasts):
    fig = go.Figure()

    # Actual Price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Ensemble Prediction
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(minutes=15),
        periods=1,
        freq='15T'
    )
    fig.add_trace(go.Scatter(
        x=future_index,
        y=[ensemble_pred_value_adjusted],
        mode='markers',
        name='Ensemble Predicted Price',
        marker=dict(color='orange', size=10)
    ))

    # Multi-step Forecasts
    if forecasts is not None:
        forecast_times = pd.date_range(
            start=df.index[-1] + pd.Timedelta(minutes=15),
            periods=len(forecasts),
            freq='15T'
        )
        fig.add_trace(go.Scatter(
            x=forecast_times,
            y=forecasts,
            mode='lines',
            name='Multi-step Forecast',
            line=dict(color='green', dash='dash')
        ))

    fig.update_layout(
        title="Price Predictions",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly",
        showlegend=True
    )
    st.plotly_chart(fig)


# ------------------------------------------------------
# 9) Main Execution
# ------------------------------------------------------
if __name__ == "__main__":
    # 1) Ticker list
    tickers = ["CC=F", "GC=F", "KC=F"]

    # 2) Streamlit UI
    st.title(
        "🚀 Enhanced Bot (CNN+LSTM+Attention with Multi-step Forecast and News Sentiment)")
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

    st.header("📊 Indices Overview")
    st.write(f"Using Ticker: {selected_ticker}")

    # 3) Fetch data
    live_data = fetch_live_data([selected_ticker])

    # 4) Fetch news articles and compute sentiment score
    st.write("Fetching news articles...")
    articles = fetch_news(selected_ticker)
    sentiment_score = compute_sentiment_score(articles)
    st.write(f"News Sentiment Score: {sentiment_score:.2f}")

    # 5) If data is returned
    if selected_ticker in live_data and not live_data[selected_ticker].empty:
        df = live_data[selected_ticker]
        st.write(f"Data for {selected_ticker}:")
        st.dataframe(df)

        # 6) Prepare & Indicators
        df = compute_indicators(df, sentiment_score)

        # 7) Enough data?
        # ...existing code...
if len(df) > 60:
    scaler = MinMaxScaler()
    missing_cols = [
        c for c in ORIGINAL_CNN_LSTM_FEATURES if c not in df.columns]
    if missing_cols:
        st.error(
            f"Missing columns required for CNN+LSTM+Attn: {missing_cols}")
    else:
        scaler.fit(df[ORIGINAL_CNN_LSTM_FEATURES].values)

        # (B) Load PyTorch models
        cnn_lstm = CNNLSTMAttention(
            n_features=len(ORIGINAL_CNN_LSTM_FEATURES))
        cnn_lstm.load_state_dict(torch.load(
            f"model/{selected_ticker}_cnn_lstm_attention.pt", map_location='cpu'))
        cnn_lstm.eval()

        transformer = TimeSeriesTransformer(
            input_size=len(ORIGINAL_CNN_LSTM_FEATURES))
        transformer.load_state_dict(torch.load(
            f"model/{selected_ticker}_transformer.pt", map_location='cpu'))
        transformer.eval()

        tcn = TCN(num_inputs=len(ORIGINAL_CNN_LSTM_FEATURES),
                  num_channels=[64, 64, 64, 64])
        tcn.load_state_dict(torch.load(
            f"model/{selected_ticker}_tcn.pt", map_location='cpu'))
        tcn.eval()

        informer = SimpleInformer(input_size=len(ORIGINAL_CNN_LSTM_FEATURES))
        informer.load_state_dict(torch.load(
            f"model/{selected_ticker}_informer.pt", map_location='cpu'))
        informer.eval()

        # (C) Prepare input data
        last_60_data = df[ORIGINAL_CNN_LSTM_FEATURES].values[-60:]
        scaled_60 = scaler.transform(last_60_data)
        X_input_deep = scaled_60.reshape(
            1, 60, len(ORIGINAL_CNN_LSTM_FEATURES))

        # (D) Ensemble Predictions
        models = [cnn_lstm, transformer, tcn, informer]

        if not models:
            st.error("No models available for prediction.")
        else:
            try:
                X_input_tensor = torch.tensor(
                    X_input_deep, dtype=torch.float32)
                preds_list = []
                for m in models:
                    with torch.no_grad():
                        pred = m(X_input_tensor).numpy().reshape(-1)
                        preds_list.append(pred)

                preds_array = np.stack(preds_list, axis=1)
                ensemble_preds = np.mean(preds_array, axis=1)
                ensemble_pred_value = ensemble_preds[0]
                ensemble_pred_value_adjusted = ensemble_pred_value * \
                    (1 + sentiment_score * 0.05)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Debug info:")
                st.write(f"Input shape: {X_input_deep.shape}")
                st.write(f"Feature columns: {ORIGINAL_CNN_LSTM_FEATURES}")
                if 'preds' in locals():
                    st.write(f"Available output keys: {list(preds.keys())}")
                raise e

        # (E) Multi-step forecast
        n_steps = 3 * 24 * 4  # Next 3 days (assuming 15-min intervals)
        forecasts = multi_step_forecast(
            cnn_lstm, X_input_deep, n_steps, scaler)
        forecast_times = pd.date_range(
            start=df.index[-1] + pd.Timedelta(minutes=15),
            periods=len(forecasts),
            freq='15T'
        )
        df_forecast = pd.DataFrame({
            'Time': forecast_times,
            'Predicted_Price': forecasts
        })
        df_forecast['Date'] = df_forecast['Time'].dt.date
        daily_ranges = df_forecast.groupby(
            'Date')['Predicted_Price'].agg(['min', 'max'])
        daily_ranges = daily_ranges.head(3)  # Next 3 days

        # (F) Determine Stop Loss and Take Profit based on current price
        current_price = df['Close'].iloc[-1]
        price_change_pct = predict_price_change_percentage(
            current_price, ensemble_pred_value_adjusted)

        if price_change_pct >= 0:
            recommended_action = 'BUY'
            stop_loss = current_price * 0.99  # 1% below current price
            take_profit = current_price * 1.02  # 2% above current price
        else:
            recommended_action = 'SELL'
            stop_loss = current_price * 1.01  # 1% above current price
            take_profit = current_price * 0.98  # 2% below current price

        confidence_level = min(
            max(abs(price_change_pct) / 2, 0), 100)

        accuracy = 85.0  # Placeholder

        # (G) Prepare Summary Data with Ensemble Predictions
        summary_data = {
            "Current Price": [current_price],
            "Ensemble Prediction": [ensemble_pred_value_adjusted],
            "Price Change (%)": [price_change_pct],
            "Recommended Action": [recommended_action],
            "Stop Loss": [stop_loss],
            "Take Profit": [take_profit]
        }

        metrics_data = {
            "Confidence Level (%)": [confidence_level],
            "Model Accuracy (%)": [accuracy]
        }

        price_predictions_df = pd.DataFrame(summary_data)
        metrics_df = pd.DataFrame(metrics_data)

        numerical_cols_price_predictions = [
            "Current Price",
            "Ensemble Prediction",
            "Price Change (%)",
            "Stop Loss",
            "Take Profit"
        ]
        numerical_cols_metrics = [
            "Confidence Level (%)",
            "Model Accuracy (%)"
        ]

        price_predictions_df[numerical_cols_price_predictions] = price_predictions_df[numerical_cols_price_predictions].astype(
            float)
        metrics_df[numerical_cols_metrics] = metrics_df[numerical_cols_metrics].astype(
            float)

        st.write("### Price Predictions:")
        st.dataframe(price_predictions_df.style.format(
            "{:.2f}", subset=numerical_cols_price_predictions))

        st.write("### Model Performance Metrics:")
        st.dataframe(metrics_df.style.format(
            "{:.2f}", subset=numerical_cols_metrics))

        st.write("### Predicted Price Ranges for Next 3 Days:")
        st.dataframe(daily_ranges.style.format("{:.2f}"))

        plot_predictions(df, ensemble_pred_value_adjusted, forecasts)
else:
    st.warning("Not enough data (need > 60 rows) to do predictions.")
