import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib  # For model persistence
from datetime import datetime, timedelta
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')  # Download the VADER lexicon for sentiment analysis

# ------------------------------------------------------
# 1) Columns / Features for CNN+LSTM+Attention
# ------------------------------------------------------
# Original features used during training
ORIGINAL_CNN_LSTM_FEATURES = [
    'Close',
    'MA_Short',
    'MA_Long',
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'Range',
    'Log_Return',
    'RSI',
    'MACD',
    'MACD_Signal',
    'MACD_Diff',
    'Bollinger_High',
    'Bollinger_Low',
    'Stoch_K',
    'Stoch_D'
]

# We can include 'News_Sentiment' in the data but not in the model input features
ALL_FEATURES = ORIGINAL_CNN_LSTM_FEATURES + ['News_Sentiment']

if 'initial_balance' not in st.session_state:
    st.session_state.initial_balance = 20000
if 'balance' not in st.session_state:
    st.session_state.balance = st.session_state.initial_balance

# ------------------------------------------------------
# 2) Custom Attention Layer (identical to training script)
# ------------------------------------------------------
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.query_dense = tf.keras.layers.Dense(128, activation='relu')
        self.key_dense = tf.keras.layers.Dense(128, activation='relu')
        self.value_dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        attention_weights = tf.nn.softmax(
            tf.matmul(query, key, transpose_b=True),
            axis=-1
        )
        weighted_sum = tf.matmul(attention_weights, inputs)
        return weighted_sum

# Load models with custom objects if necessary
def load_keras_model(ticker, model_name):
    model_path = os.path.join("model", f"{ticker}_{model_name}.h5")
    if model_name == 'cnn_lstm':
        loaded_model = keras_load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
    else:
        loaded_model = keras_load_model(model_path)
    return loaded_model

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
                    st.warning(f"Retrying for {ticker}... Attempt {attempt + 1}")
                else:
                    st.error(f"Failed to fetch data for {ticker}: {e}")
        else:
            st.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return data

# ------------------------------------------------------
# 4) News Sentiment Analysis
# ------------------------------------------------------
def fetch_news(ticker):
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    if not news_api_key:
        st.error("API key not found for NewsAPI.org.")
        return []

    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={news_api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        st.error("Failed to fetch news articles.")
        return []

def compute_sentiment_score(articles):
    # Handle cases with no articles
    if not articles:
        # No articles found, return neutral sentiment
        return 0.0

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
        average_score = 0.0
    return average_score

def get_article_summaries(articles):
    summaries = []
    for article in articles:
        title = article.get('title', '')
        url = article.get('url', '')
        published_at = article.get('publishedAt', '')
        summaries.append({
            'Title': title,
            'URL': url,
            'Published At': published_at
        })
    return summaries

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
    if current_price == 0 or predicted_price is None or predicted_price == 0:
        return 0.0
    return ((predicted_price - current_price) / current_price) * 100

def compute_indicators(df, sentiment_score):
    df = df.copy()
    if 'Close' not in df.columns:
        st.warning("Data does not contain 'Close' column.")
        return df

    df = prepare_data(df, sentiment_score)  # This function already computes all needed indicators

    return df

def multi_step_forecast(model, X_last, n_steps, scaler):
    """
    Performs recursive multi-step forecasting using the last available data.
    """
    forecasts = []
    X_input = X_last.copy()  # shape (1, time_steps, n_features)
    for _ in range(n_steps):
        pred = model.predict(X_input)
        forecasts.append(pred[0, 0])
        # Prepare next input by appending prediction and removing oldest data
        last_features = X_input[0, 1:, :]  # Skip the first time step
        next_feature = X_input[0, -1:, :].copy()  # Copy of last time step
        next_feature[0, 0] = pred[0, 0]  # Update 'Close' with prediction
        X_input = np.concatenate([last_features, next_feature], axis=0)
        X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])
    # Inverse transform forecasts
    inv_forecasts = []
    for pred in forecasts:
        dummy = np.zeros((1, len(ORIGINAL_CNN_LSTM_FEATURES)))  # Use original features length
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
        freq='15min'
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
            freq='15min'
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
    tickers = ["CC=F", "GC=F", "KC=F", "NG=F", "^GDAXI", "^HSI", "USD/JPY", "ETHUSD", "SOLUSD", "^SPX", "HG=F", "SI=F", "CL=F", "^VIX", "ACB", "CGC", "CL=F", "TLRY"]

    # 2) Streamlit UI
    st.title("Enhanced Bot (CNN+LSTM+Attention with Multi-step Forecast and News Sentiment)")
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

    st.header("Indices Overview")
    st.write(f"Using Ticker: {selected_ticker}")

    # 3) Fetch data
    live_data = fetch_live_data([selected_ticker])

    # 4) Fetch news articles and compute sentiment score
    st.write("Fetching news articles...")
    articles = fetch_news(selected_ticker)
    sentiment_score = compute_sentiment_score(articles)
    st.write(f"News Sentiment Score: {sentiment_score:.2f}")

    # Display news articles
    st.write("### Latest News Articles:")
    article_summaries = get_article_summaries(articles)
    if article_summaries:
        news_df = pd.DataFrame(article_summaries)
        st.dataframe(news_df)
    else:
        st.write("No recent news articles found.")

    # 5) If data is returned
    if selected_ticker in live_data and not live_data[selected_ticker].empty:
        df = live_data[selected_ticker]
        st.write(f"Data for {selected_ticker}:")
        st.dataframe(df)

        # 6) Prepare & Indicators
        df = compute_indicators(df, sentiment_score)

        # 7) Enough data?
        if len(df) > 60:
            # (A) Fit a scaler on the features used by CNN+LSTM
            scaler = MinMaxScaler()
            missing_cols = [c for c in ORIGINAL_CNN_LSTM_FEATURES if c not in df.columns]
            if missing_cols:
                st.error(f"Missing columns required for CNN+LSTM+Attn: {missing_cols}")
            else:
                # Ensure correct column order
                df_model_features = df[ORIGINAL_CNN_LSTM_FEATURES]
                scaler.fit(df_model_features.values)

                # (B) Load the models
                # Load CNN+LSTM+Attention model
                try:
                    loaded_cnn_lstm_model = load_keras_model(selected_ticker, 'cnn_lstm')
                    st.success(f"Loaded CNN+LSTM+Attention model for {selected_ticker}.")
                except Exception as e:
                    st.error(f"Failed to load CNN+LSTM+Attention model for {selected_ticker}: {e}")
                    loaded_cnn_lstm_model = None

                # Load other models
                try:
                    transformer = load_keras_model(selected_ticker, 'transformer')
                    st.success("Loaded Transformer model.")
                except Exception as e:
                    st.error(f"Failed to load Transformer model: {e}")
                    transformer = None

                try:
                    tcn = load_keras_model(selected_ticker, 'tcn')
                    st.success("Loaded TCN model.")
                except Exception as e:
                    st.error(f"Failed to load TCN model: {e}")
                    tcn = None

                try:
                    informer = load_keras_model(selected_ticker, 'informer')
                    st.success("Loaded Informer model.")
                except Exception as e:
                    st.error(f"Failed to load Informer model: {e}")
                    informer = None

                # (C) Prepare input data
                last_60_data = df_model_features.values[-60:]  # shape (60, len(ORIGINAL_CNN_LSTM_FEATURES))
                scaled_60 = scaler.transform(last_60_data)     # shape (60, len(ORIGINAL_CNN_LSTM_FEATURES))
                X_input_deep = scaled_60.reshape(1, 60, len(ORIGINAL_CNN_LSTM_FEATURES))  # shape (1,60,len(ORIGINAL_CNN_LSTM_FEATURES))

                # (D) Ensemble Predictions
                models = [loaded_cnn_lstm_model, transformer, tcn, informer]
                models = [m for m in models if m is not None]  # Remove any models that failed to load

                if not models:
                    st.error("No models available for prediction.")
                else:
                    # Average predictions from available models
                    preds_list = []
                    for idx, m in enumerate(models):
                        p = m.predict(X_input_deep).reshape(-1)
                        st.write(f"Model {idx+1} Prediction: {p[0]}")  # Output individual model predictions
                        preds_list.append(p)
                    # Stack predictions
                    preds_array = np.stack(preds_list, axis=1)  # shape: (num_samples, num_models)
                    ensemble_preds = np.mean(preds_array, axis=1)     # shape: (num_samples,)

                    ensemble_pred_value = ensemble_preds[0]

                    # Adjust prediction based on sentiment
                    sentiment_adjustment_factor = 0.02  # Adjust as needed
                    ensemble_pred_value_adjusted = ensemble_pred_value * (1 + sentiment_score * sentiment_adjustment_factor)

                    # (E) Multi-step forecast
                    n_steps = 3 * 24 * 4  # Next 3 days (assuming 15-min intervals)
                    forecasts = multi_step_forecast(loaded_cnn_lstm_model, X_input_deep, n_steps, scaler)
                    forecast_times = pd.date_range(
                        start=df.index[-1] + pd.Timedelta(minutes=15),
                        periods=len(forecasts),
                        freq='15min'
                    )
                    df_forecast = pd.DataFrame({
                        'Time': forecast_times,
                        'Predicted_Price': forecasts
                    })
                    df_forecast['Date'] = df_forecast['Time'].dt.date
                    daily_ranges = df_forecast.groupby('Date')['Predicted_Price'].agg(['min', 'max'])
                    daily_ranges = daily_ranges.head(3)  # Next 3 days

                    # (F) Determine Stop Loss and Take Profit based on current price
                    current_price = df['Close'].iloc[-1]
                    price_change_pct = predict_price_change_percentage(current_price, ensemble_pred_value_adjusted)

                    # Debugging statements
                    st.write(f"Current Price: {current_price}")
                    st.write(f"Ensemble Predicted Price (before adjustment): {ensemble_pred_value}")
                    st.write(f"Sentiment Score: {sentiment_score}")
                    st.write(f"Sentiment Adjustment Factor: {sentiment_adjustment_factor}")
                    st.write(f"Ensemble Predicted Price (after adjustment): {ensemble_pred_value_adjusted}")
                    st.write(f"Predicted Price Change Percentage: {price_change_pct}%")

                    # Define a threshold for HOLD signal
                    hold_threshold = 1.0  # Threshold percentage for HOLD signal

                    if price_change_pct >= hold_threshold:
                        # Predicted price is significantly higher than current price, recommend to BUY
                        recommended_action = 'BUY'
                        stop_loss = current_price * 0.99  # 1% below current price
                        take_profit = current_price * 1.02  # 2% above current price
                    elif price_change_pct <= -hold_threshold:
                        # Predicted price is significantly lower, recommend to SELL
                        recommended_action = 'SELL'
                        stop_loss = current_price * 1.01  # 1% above current price
                        take_profit = current_price * 0.98  # 2% below current price
                    else:
                        # Predicted price change is insignificant, recommend to HOLD
                        recommended_action = 'HOLD'
                        stop_loss = None
                        take_profit = None

                    st.write(f"Recommended Action: {recommended_action}")

                    # Estimate confidence level based on magnitude of price change
                    confidence_level = min(max(abs(price_change_pct) / 2, 0), 100)  # Scaled between 0% and 100%

                    # Placeholder for accuracy (should be replaced with actual model accuracy)
                    accuracy = 85.0  # Assuming model accuracy is 85%

                    # (G) Potential Profit Screener
                    potential_profit = None
                    if recommended_action == 'BUY':
                        potential_profit = (take_profit - current_price) * 1  # Assuming 1 unit purchase
                    elif recommended_action == 'SELL':
                        potential_profit = (current_price - take_profit) * 1  # Assuming 1 unit sale

                    # (H) Prepare Summary Data with Ensemble Predictions
                    summary_data = {
                        "Current Price": [current_price],
                        "Ensemble Prediction": [ensemble_pred_value_adjusted],
                        "Price Change (%)": [price_change_pct],
                        "Recommended Action": [recommended_action],
                        "Stop Loss": [stop_loss if stop_loss is not None else "N/A"],
                        "Take Profit": [take_profit if take_profit is not None else "N/A"],
                        "Potential Profit": [potential_profit if potential_profit is not None else "N/A"]
                    }

                    metrics_data = {
                        "Confidence Level (%)": [confidence_level],
                        "Model Accuracy (%)": [accuracy]
                    }

                    # Create DataFrames
                    price_predictions_df = pd.DataFrame(summary_data)
                    metrics_df = pd.DataFrame(metrics_data)

                    # List of numerical columns for formatting
                    numerical_cols_price_predictions = [
                        "Current Price",
                        "Ensemble Prediction",
                        "Price Change (%)",
                        "Stop Loss",
                        "Take Profit",
                        "Potential Profit"
                    ]
                    numerical_cols_metrics = [
                        "Confidence Level (%)",
                        "Model Accuracy (%)"
                    ]

                    # Ensure numerical columns are of type float
                    price_predictions_df[numerical_cols_price_predictions] = price_predictions_df[numerical_cols_price_predictions].astype(float, errors='ignore')
                    metrics_df[numerical_cols_metrics] = metrics_df[numerical_cols_metrics].astype(float)

                    # Display Price Predictions Table with formatting applied only to numerical columns
                    st.write("### Price Predictions:")
                    st.dataframe(price_predictions_df.style.format("{:.2f}", subset=numerical_cols_price_predictions))

                    # Display Metrics Table with formatting applied only to numerical columns
                    st.write("### Model Performance Metrics:")
                    st.dataframe(metrics_df.style.format("{:.2f}", subset=numerical_cols_metrics))

                    # Display Price Ranges
                    st.write("### Predicted Price Ranges for Next 3 Days:")
                    st.dataframe(daily_ranges.style.format("{:.2f}"))

                    # (I) Plot
                    plot_predictions(df, ensemble_pred_value_adjusted, forecasts)
        else:
            st.warning("Not enough data (need > 60 rows) to do predictions.")

    elif selected_ticker not in live_data or live_data[selected_ticker].empty:
        st.warning("No data available for this ticker. Cannot proceed.")