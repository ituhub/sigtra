import torch
import torch.nn as nn
import torch.optim as optim
import os
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

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

os.makedirs("model", exist_ok=True)

# --------------------------------------------------------------------------------
# 1) Data Fetching
# --------------------------------------------------------------------------------


def fetch_live_data(tickers, retries=3):
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set FMP_API_KEY in your environment.")
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
                df.sort_index(inplace=True)
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
    df['MA_Short'] = df['Close'].rolling(window=5).mean()
    df['MA_Long'] = df['Close'].rolling(window=20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Range'] = df['High'] - df['Low']
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    if TA_LIB_AVAILABLE:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        boll = ta.volatility.BollingerBands(
            df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = boll.bollinger_hband()
        df['Bollinger_Low'] = boll.bollinger_lband()
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], 14)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

# --------------------------------------------------------------------------------
# 3) Data Prep for Deep Models
# --------------------------------------------------------------------------------


def prepare_lstm_data(df, feature_cols, time_step=60):
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


def prepare_torch_data(X, y, batch_size=64):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    return loader

# --------------------------------------------------------------------------------
# 4) PyTorch Model Definitions
# --------------------------------------------------------------------------------

# Attention block


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)))
        out = torch.matmul(attn_weights, V)
        return out

# CNN-LSTM-Attention


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features, filters=64, kernel_size=3, lstm_units=100, attention_dim=128, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, filters,
                               kernel_size, padding='same')
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(filters, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(lstm_units, attention_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(attention_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, filters)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)

# TCN


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(
            in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Match lengths before residual addition
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers += [TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(out_ch, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x.squeeze(-1)


# Transformer


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze(-1)

# Informer (simplified)


class SimpleInformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze(-1)

# Autoencoder


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, seq_len, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * input_size)
        )
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(x.size(0), self.seq_len, self.input_size)
        return out

# --------------------------------------------------------------------------------
# 5) Training Loop
# --------------------------------------------------------------------------------


def train_pytorch_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
    return model

# --------------------------------------------------------------------------------
# 6) Multi-step Forecasting (for PyTorch models)
# --------------------------------------------------------------------------------


def multi_step_forecast(model, X_last, n_steps, scaler, feature_cols, device='cpu'):
    model.eval()
    forecasts = []
    X_input = torch.tensor(X_last, dtype=torch.float32).to(device)
    for _ in range(n_steps):
        with torch.no_grad():
            pred = model(X_input).cpu().numpy()
        forecasts.append(pred[0])
        X_np = X_input.cpu().numpy()
        X_np = np.roll(X_np, -1, axis=1)
        X_np[0, -1, feature_cols.index('Close')] = pred[0]
        X_input = torch.tensor(X_np, dtype=torch.float32).to(device)
    inv_forecasts = []
    for pred in forecasts:
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, feature_cols.index('Close')] = pred
        inv_pred = scaler.inverse_transform(dummy)
        inv_close = inv_pred[0, feature_cols.index('Close')]
        inv_forecasts.append(inv_close)
    return inv_forecasts

# --------------------------------------------------------------------------------
# 7) XGBoost Data Prep and Training
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    print(f"XGBoost test MSE: {mse:.6f}")
    return model


# --------------------------------------------------------------------------------
# 8) Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["NG=F"]  # Use a single ticker for testing
    data = fetch_live_data(tickers)
    for ticker in tickers:
        df = data.get(ticker, pd.DataFrame())
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        df = enhance_features(df)
        if len(df) < 60:
            print(f"Not enough rows after feature eng. Rows: {len(df)}")
            continue

        base_cols = ["Close", "MA_Short", "MA_Long",
                     "Lag_1", "Lag_2", "Lag_3", "Range", "Log_Return"]
        adv_cols = []
        if TA_LIB_AVAILABLE:
            adv_cols = ["RSI", "MACD", "MACD_Signal", "MACD_Diff",
                        "Bollinger_High", "Bollinger_Low", "Stoch_K", "Stoch_D"]
        feature_cols = [c for c in base_cols + adv_cols if c in df.columns]

        X_ts, y_ts, scaler = prepare_lstm_data(df, feature_cols, time_step=60)
        split = int(0.8 * len(X_ts))
        X_train, X_test = X_ts[:split], X_ts[split:]
        y_train, y_test = y_ts[:split], y_ts[split:]

        train_loader = prepare_torch_data(X_train, y_train, batch_size=64)
        val_loader = prepare_torch_data(X_test, y_test, batch_size=64)

        # CNN-LSTM-Attention
        print("\n--- Training CNN-LSTM-Attention ---")
        cnn_lstm = CNNLSTMAttention(n_features=len(feature_cols))
        cnn_lstm = train_pytorch_model(
            cnn_lstm, train_loader, val_loader, epochs=20, lr=0.001)
        torch.save(cnn_lstm.state_dict(),
                   f"model/{ticker}_cnn_lstm_attention.pt")

        # Transformer
        print("\n--- Training Transformer ---")
        transformer = TimeSeriesTransformer(input_size=len(feature_cols))
        transformer = train_pytorch_model(
            transformer, train_loader, val_loader, epochs=20, lr=0.001)
        torch.save(transformer.state_dict(), f"model/{ticker}_transformer.pt")

        # TCN
        print("\n--- Training TCN ---")
        tcn = TCN(num_inputs=len(feature_cols), num_channels=[64, 64, 64, 64])
        tcn = train_pytorch_model(
            tcn, train_loader, val_loader, epochs=20, lr=0.001)
        torch.save(tcn.state_dict(), f"model/{ticker}_tcn.pt")

        # Informer (simplified)
        print("\n--- Training Informer (Simplified) ---")
        informer = SimpleInformer(input_size=len(feature_cols))
        informer = train_pytorch_model(
            informer, train_loader, val_loader, epochs=20, lr=0.001)
        torch.save(informer.state_dict(), f"model/{ticker}_informer.pt")

        # Autoencoder (optional)
        print("\n--- Training Autoencoder ---")
        autoencoder = TimeSeriesAutoencoder(
            input_size=len(feature_cols), seq_len=60)
        autoencoder = train_pytorch_model(
            autoencoder, train_loader, val_loader, epochs=20, lr=0.001)
        torch.save(autoencoder.state_dict(), f"model/{ticker}_autoencoder.pt")

        # XGBoost (optional)
        if XGBOOST_AVAILABLE:
            print("\n--- XGBoost ---")
            df_xgb = df.copy()
            xgb_data, xgb_labels = prepare_xgb_data(
                df_xgb, feature_cols, target_col='Close', forward_shift=1)
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                xgb_data, xgb_labels, test_size=0.2, shuffle=False)
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
            xgb_model.fit(X_train_xgb, y_train_xgb)
            preds = xgb_model.predict(X_test_xgb)
            mse = np.mean((preds - y_test_xgb) ** 2)
            print(f"XGBoost test MSE: {mse:.6f}")
            joblib.dump(xgb_model, f"model/{ticker}_xgb.pkl")

        # Multi-step forecast example (using CNN-LSTM)
        print("\n--- Multi-step Forecast for Next 3 Days ---")
        n_steps = 3 * 24 * 4  # 3 days * 24 hours * 4 intervals per hour
        forecasts = multi_step_forecast(
            cnn_lstm, X_ts[-1:], n_steps, scaler, feature_cols)
        df_forecast = pd.DataFrame({
            'Time': pd.date_range(start=df.index[-1], periods=n_steps, freq='15T'),
            'Predicted_Price': forecasts
        })
        df_forecast['Date'] = df_forecast['Time'].dt.date
        daily_ranges = df_forecast.groupby(
            'Date')['Predicted_Price'].agg(['min', 'max'])
        print("Predicted Price Ranges for Next 3 Days:")
        print(daily_ranges.head(3))

        print(f"\n--- Finished {ticker} ---")
