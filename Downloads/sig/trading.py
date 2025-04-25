import os
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Input,
    LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, Layer
)
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch

# Optional advanced TA features
try:
    import ta
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False

# For ARIMA or other advanced classical time-series
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# For XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --------------------------------------------------------------------------------
# 1) Data Fetching
# --------------------------------------------------------------------------------
def fetch_live_data(tickers, retries=3):
    """
    Fetch 15-minute historical data from FinancialModelingPrep using FMP_API_KEY.
    """
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set FMP_API_KEY in your environment.")

    for ticker in tickers:
        for attempt in range(retries):
            try:
                ticker_api = ticker.replace('/', '')
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                if not data_json or len(data_json) < 1:
                    continue

                df = pd.DataFrame(data_json)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.rename(
                    columns={
                        'close': 'Close',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'volume': 'Volume'
                    },
                    inplace=True
                )
                df.sort_index(inplace=True)  # ascending
                data[ticker] = df
                break
            except Exception as e:
                if attempt < retries - 1:
                    continue
                else:
                    raise e
    return data


# --------------------------------------------------------------------------------
# 2) Feature Engineering
# --------------------------------------------------------------------------------
def enhance_features(df):
    """
    Adds MAs, lags, log returns, range, and optional advanced TA features.
    """
    df['MA_Short'] = df['Close'].rolling(window=5).mean()
    df['MA_Long'] = df['Close'].rolling(window=20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)

    df['Range'] = df['High'] - df['Low']
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    if TA_LIB_AVAILABLE:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        # Bollinger Bands
        boll = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = boll.bollinger_hband()
        df['Bollinger_Low'] = boll.bollinger_lband()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 14)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------------
# 3) Data Prep for Deep Models
# --------------------------------------------------------------------------------
def prepare_lstm_data(df, feature_cols, time_step=60):
    """
    Produces (X, y) in shape:
      X: (num_samples, time_step=60, num_features),
      y: (num_samples,).
    """
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    close_idx = feature_cols.index("Close")
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, :])
        y.append(scaled_data[i, close_idx])
    X, y = np.array(X), np.array(y)
    return X, y, scaler


# --------------------------------------------------------------------------------
# 4) Custom Attention Layer
# --------------------------------------------------------------------------------
class AttentionLayer(Layer):
    def __init__(self):
        super().__init__()
        self.query_dense = tf.keras.layers.Dense(128, activation='relu')
        self.key_dense = tf.keras.layers.Dense(128, activation='relu')
        self.value_dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True), axis=-1)
        weighted_sum = tf.matmul(attention_weights, inputs)
        return weighted_sum


# --------------------------------------------------------------------------------
# 5) Model Definitions
# --------------------------------------------------------------------------------
def build_cnn_lstm_attention_model(
    filters=64,
    kernel_size=3,
    lstm_units=100,
    dropout_rate=0.3,
    learning_rate=0.001,
    n_features=6
):
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(60, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(AttentionLayer())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model


def build_transformer_model(
    d_model=32,
    num_heads=2,
    ff_dim=64,
    dropout_rate=0.1,
    sequence_length=60,
    n_features=6,
    learning_rate=0.001
):
    """
    Builds a simplified Transformer-based model for time-series forecasting.
    Ensures each layer is called on a tensor (avoiding the 'Got <Dropout> object' error).
    """

    # 1) Inputs
    inputs = Input(shape=(sequence_length, n_features))

    # 2) Dense projection to d_model dimension
    x = Dense(d_model)(inputs)

    # 3) Optional dropout on that projection
    x = Dropout(dropout_rate)(x)

    # 4) Layer Normalization
    x = LayerNormalization(epsilon=1e-6)(x)

    # 5) Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    # Residual
    x = x + attn_output

    # 6) Another layer norm
    x = LayerNormalization(epsilon=1e-6)(x)

    # 7) Position-wise feed-forward sublayer
    ff = Sequential([
        Dense(ff_dim, activation='relu'),
        Dropout(dropout_rate),
        Dense(d_model),
    ])

    # 8) Residual around the feed-forward
    x = x + ff(x)

    # 9) Another layer norm
    x = LayerNormalization(epsilon=1e-6)(x)

    # 10) Pool over time dimension
    x = GlobalAveragePooling1D()(x)

    # 11) Final dropout
    x = Dropout(dropout_rate)(x)

    # 12) Output for regression (forecast)
    outputs = Dense(1)(x)

    # Build model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model


def build_tcn_model(
    filters=64,
    kernel_size=3,
    dilation_rates=(1, 2, 4, 8),
    dropout_rate=0.2,
    learning_rate=0.001,
    n_features=6
):
    input_layer = Input(shape=(60, n_features))
    x = input_layer
    for d in dilation_rates:
        x_skip = x
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d,
            padding='causal',
            activation='relu'
        )(x)
        x = Dropout(dropout_rate)(x)
        # Residual
        if x_skip.shape[-1] != x.shape[-1]:
            x_skip = Conv1D(filters, 1, padding='same')(x_skip)
        x = x + x_skip
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(input_layer, outputs)
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model


def build_informer_model(
    d_model=32,
    num_heads=2,
    dropout_rate=0.1,
    sequence_length=60,
    n_features=6,
    learning_rate=0.001
):
    """
    Simplified Informer-like block.
    """
    inputs = Input(shape=(sequence_length, n_features))
    x = Dense(d_model)(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = x + attn_output
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Sequential([Dense(d_model, activation='relu'), Dense(d_model)])
    x = x + ff(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model


# --------------------------------------------------------------------------------
# 6) Hyperparameter Tuning (CNN-LSTM)
# --------------------------------------------------------------------------------
def tune_model(X, y, n_features=6):
    """
    Hyperparameter tuning for CNN-LSTM-Attention using keras_tuner.
    """
    def model_builder(hp):
        return build_cnn_lstm_attention_model(
            filters=hp.Int("filters", 32, 128, step=16),
            kernel_size=hp.Choice("kernel_size", [3, 5]),
            lstm_units=hp.Int("lstm_units", 50, 200, step=50),
            dropout_rate=hp.Choice("dropout_rate", [0.2, 0.3, 0.5]),
            learning_rate=hp.Choice("learning_rate", [0.001, 0.0001]),
            n_features=n_features
        )

    tuner = RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='cnn_lstm_attention_tuning'
    )

    tuner.search(
        X, y,
        epochs=5,
        validation_split=0.2,
        shuffle=False,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
    )
    return tuner


# --------------------------------------------------------------------------------
# 7) Cross-Validation
# --------------------------------------------------------------------------------
def cross_validate_model(X, y, model_builder, num_folds=3, epochs=5, batch_size=64):
    tscv = TimeSeriesSplit(n_splits=num_folds)
    results = []
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        model = model_builder()
        model.fit(X_train_cv, y_train_cv, epochs=epochs, batch_size=batch_size,
                  verbose=1, shuffle=False)
        loss = model.evaluate(X_val_cv, y_val_cv, verbose=0)
        results.append(loss)
    print("\nTimeSeriesSplit results:", results)
    print("Average loss:", np.mean(results))


# --------------------------------------------------------------------------------
# 8) ARIMA (classical)
# --------------------------------------------------------------------------------
def train_arima(df, target_col='Close', order=(1, 1, 1)):
    if not STATSMODELS_AVAILABLE:
        print("statsmodels not installed. Skipping ARIMA.")
        return None

    series = df[target_col].dropna().values
    split_idx = int(0.8 * len(series))
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    model = ARIMA(train_data, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_data))
    mse = np.mean((forecast - test_data) ** 2)
    print(f"ARIMA (order={order}) MSE on test set: {mse:.6f}")
    return fitted


# --------------------------------------------------------------------------------
# 9) XGBoost
# --------------------------------------------------------------------------------
def prepare_xgb_data(df, feature_cols, target_col='Close', forward_shift=1):
    df['Target'] = df[target_col].shift(-forward_shift)
    df.dropna(inplace=True)
    X = df[feature_cols].values
    y = df['Target'].values
    return X, y

def train_xgb_model(X, y):
    if not XGBOOST_AVAILABLE:
        print("xgboost not installed. Skipping XGBoost.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    print(f"XGBoost test MSE: {mse:.6f}")
    return model


# --------------------------------------------------------------------------------
# 10) Ensemble Predictions
# --------------------------------------------------------------------------------
def ensemble_predict(models, X):
    """
    Average predictions from multiple Keras models that share the same input shape.
    """
    preds_list = []
    for m in models:
        p = m.predict(X)  # shape: (num_samples, 1)
        p = p.reshape(-1)  # flatten to (num_samples,)
        preds_list.append(p)
    # Each array in preds_list has shape (num_test_samples,)
    preds_array = np.stack(preds_list, axis=1)  # (num_test_samples, num_models)
    ensemble = np.mean(preds_array, axis=1)     # (num_test_samples,)
    return ensemble


# --------------------------------------------------------------------------------
# 11) Simple Take-Profit/Stop-Loss Simulation
# --------------------------------------------------------------------------------
def simulate_trades_with_sl_tp(predictions, actual_prices, take_profit=0.02, stop_loss=0.01):
    if len(predictions) != len(actual_prices):
        raise ValueError("predictions vs. actual_prices size mismatch.")

    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    trades = []
    in_position = False
    entry_price = None

    for i in range(len(predictions) - 1):
        if not in_position:
            # If predicted next step is higher => buy
            if predictions[i+1] > actual_prices[i]:
                in_position = True
                entry_price = actual_prices[i+1]
                trades.append(("BUY", i+1, entry_price))
        else:
            current_price = actual_prices[i+1]
            change = (current_price - entry_price) / entry_price
            if change >= take_profit:
                trades.append(("SELL_TP", i+1, current_price))
                in_position = False
            elif change <= -stop_loss:
                trades.append(("SELL_SL", i+1, current_price))
                in_position = False

    if in_position:
        # close any open position at end
        final_price = actual_prices[-1]
        trades.append(("SELL_END", len(predictions) - 1, final_price))

    return trades


# --------------------------------------------------------------------------------
# 12) Multi-step Forecasting
# --------------------------------------------------------------------------------
def recursive_forecast(model, X_last, n_steps, scaler, feature_cols):
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
        next_feature = np.zeros((1, 1, X_input.shape[2]))
        next_feature[0, 0, :] = X_input[0, -1, :]
        # Update 'Close' with prediction
        close_idx = feature_cols.index('Close')
        next_feature[0, 0, close_idx] = pred[0, 0]
        X_input = np.concatenate([last_features, next_feature[0]], axis=0)
        X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])

    # Inverse transform forecasts
    inv_forecasts = []
    for pred in forecasts:
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, feature_cols.index('Close')] = pred
        inv_pred = scaler.inverse_transform(dummy)
        inv_close = inv_pred[0, feature_cols.index('Close')]
        inv_forecasts.append(inv_close)
    return inv_forecasts


# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["CC=F", "GC=F", "KC=F", "NG=F", "^GDAXI", "^HSI", "USD/JPY", "ETHUSD", "SOLUSD", "^SPX", "HG=F", "SI=F", "CL=F", "^VIX", "ACB", "CGC", "CL=F", "TLRY"]

    data = fetch_live_data(tickers)
    for ticker in tickers:
        df = data.get(ticker, pd.DataFrame())
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        print(f"\n--- Ticker: {ticker} ---")
        print(f"Rows: {len(df)}")

        # 1) Enhance
        df = enhance_features(df)
        if len(df) < 60:
            print(f"Not enough rows after feature eng. Rows: {len(df)}")
            continue

        # Basic columns + advanced columns (if TA installed)
        base_cols = ["Close", "MA_Short", "MA_Long", "Lag_1", "Lag_2", "Lag_3", "Range", "Log_Return"]
        adv_cols = []
        if TA_LIB_AVAILABLE:
            adv_cols = ["RSI", "MACD", "MACD_Signal", "MACD_Diff",
                        "Bollinger_High", "Bollinger_Low", "Stoch_K", "Stoch_D"]
        feature_cols = [c for c in base_cols + adv_cols if c in df.columns]

        # 2) Prepare data for deep learning (LSTM shape)
        X_ts, y_ts, scaler = prepare_lstm_data(df, feature_cols, time_step=60)
        print(f"LSTM data shapes: X={X_ts.shape}, y={y_ts.shape}")

        # 3) Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, shuffle=False)
        print(f"Train: {X_train.shape}, {y_train.shape}; Test: {X_test.shape}, {y_test.shape}")

        # 4) CNN-LSTM Tuning
        print("\n--- Tuning CNN-LSTM-Attention ---")
        tuner = tune_model(X_train, y_train, n_features=len(feature_cols))
        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_cnn_lstm = tuner.hypermodel.build(best_hp)
        best_cnn_lstm.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=False)
        cnn_loss = best_cnn_lstm.evaluate(X_test, y_test)
        print(f"Best CNN-LSTM test loss: {cnn_loss:.6f}")

        # 5) Transformer
        print("\n--- Transformer ---")
        transformer = build_transformer_model(
            d_model=32, num_heads=2, ff_dim=64, dropout_rate=0.1,
            sequence_length=60, n_features=len(feature_cols), learning_rate=0.001
        )
        transformer.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=False)
        t_loss = transformer.evaluate(X_test, y_test)
        print(f"Transformer test loss: {t_loss:.6f}")

        # 6) TCN
        print("\n--- TCN ---")
        tcn = build_tcn_model(
            filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8],
            dropout_rate=0.2, learning_rate=0.001, n_features=len(feature_cols)
        )
        tcn.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=False)
        tcn_loss = tcn.evaluate(X_test, y_test)
        print(f"TCN test loss: {tcn_loss:.6f}")

    # 8) Informer
        print("\n--- Informer (Simplified) ---")
        informer = build_informer_model(
            d_model=32, num_heads=2, dropout_rate=0.1,
            sequence_length=60, n_features=len(feature_cols), learning_rate=0.001
        )
        informer.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=False)
        inf_loss = informer.evaluate(X_test, y_test)
        print(f"Informer test loss: {inf_loss:.6f}")

        # 9) XGBoost (optional if installed)
        xgb_model = None
        if XGBOOST_AVAILABLE:
            print("\n--- XGBoost ---")
            # Prepare data for XGBoost
            df_xgb = df.copy()
            xgb_data, xgb_labels = prepare_xgb_data(df_xgb, feature_cols, target_col='Close', forward_shift=1)
            xgb_model = train_xgb_model(xgb_data, xgb_labels)

        # 10) Save all deep models
        os.makedirs("model", exist_ok=True)
        best_cnn_lstm.save(f"model/{ticker}_cnn_lstm.h5")
        transformer.save(f"model/{ticker}_transformer.h5")
        tcn.save(f"model/{ticker}_tcn.h5")
        informer.save(f"model/{ticker}_informer.h5")

        # Save XGBoost if available
        if xgb_model is not None:
            joblib.dump(xgb_model, f"model/{ticker}_xgb.pkl")

        # 11) Ensemble with all the deep models
        ensemble_models = [best_cnn_lstm, transformer, tcn, informer]
        for m in ensemble_models:
            shape_pred = m.predict(X_test).shape
            print(f"{m.name} => predict shape={shape_pred}")

        ensemble_preds = ensemble_predict(ensemble_models, X_test)
        mse_ensemble = np.mean((ensemble_preds - y_test) ** 2)
        print(f"Ensemble test MSE: {mse_ensemble:.6f}")

        # Simulate trades
        print("\n--- Simulating Trades with Stop-Loss and Take-Profit ---")

        # Adjusted to take the last len(y_test) Close prices
        actual_test_prices = df['Close'].values[-len(y_test):]

        # Verify lengths
        print(f"Length of ensemble_preds: {len(ensemble_preds)}")
        print(f"Length of actual_test_prices: {len(actual_test_prices)}")

        trades = simulate_trades_with_sl_tp(
            predictions=ensemble_preds,
            actual_prices=actual_test_prices,
            take_profit=0.02,
            stop_loss=0.01
        )
        print("Trades:", trades)

        # 13) Multi-step forecast for next 3 days (15-min intervals)
        print("\n--- Multi-step Forecast for Next 3 Days ---")
        n_steps = 3 * 24 * 4  # 3 days * 24 hours * 4 intervals per hour
        # Use best CNN-LSTM to forecast
        forecasts = recursive_forecast(best_cnn_lstm, X_ts[-1:], n_steps, scaler, feature_cols)
        df_forecast = pd.DataFrame({
            'Time': pd.date_range(start=df.index[-1], periods=n_steps, freq='15T'),
            'Predicted_Price': forecasts
        })
        df_forecast['Date'] = df_forecast['Time'].dt.date
        daily_ranges = df_forecast.groupby('Date')['Predicted_Price'].agg(['min', 'max'])
        print("Predicted Price Ranges for Next 3 Days:")
        print(daily_ranges.head(3))

        print(f"\n--- Finished {ticker} ---")
