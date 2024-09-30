import streamlit as st
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from dotenv import load_dotenv

# Configuration
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD",
                  "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD"]

# Initialize session state
if 'balance' not in st.session_state:
    st.session_state.balance = 100000
if 'positions' not in st.session_state:
    st.session_state.positions = {}  # {'symbol': {'quantity': ..., 'cost_basis': ...}}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = {}
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = [
        {'Date': datetime.now(), 'Balance': 100000}]


def get_data(symbol, start_date, end_date, source="yfinance"):
    if source == "yfinance":
        data = yf.download(symbol, start=start_date, end=end_date)
    else:  # Alpha Vantage
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.loc[start_date:end_date]
    return data


def calculate_signals(data):
    # Ensure data is sorted by date
    data = data.sort_index()

    # Add all TA features
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Strategy 1: SMA Crossover
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['Signal_SMA'] = np.where(data['SMA20'] > data['SMA50'], 1, -1)

    # Strategy 2: RSI
    rsi = RSIIndicator(data['Close'])
    data['RSI'] = rsi.rsi()
    data['Signal_RSI'] = np.where(
        data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))

    # Strategy 3: MACD
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['Signal_Line'] = macd.macd_signal()
    data['Signal_MACD'] = np.where(data['MACD'] > data['Signal_Line'], 1, -1)

    # Strategy 4: Bollinger Bands
    bb = BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['Signal_BB'] = np.where(data['Close'] < data['BB_Low'], 1, np.where(
        data['Close'] > data['BB_High'], -1, 0))

    # Strategy 5: Stochastic Oscillator
    stoch = StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['Signal_Stoch'] = np.where(
        (data['Stoch_K'] < 20) & (data['Stoch_K'] > data['Stoch_D']), 1,
        np.where((data['Stoch_K'] > 80) & (data['Stoch_K'] < data['Stoch_D']), -1, 0))

    # Strategy 6: EMA Ribbon
    data['EMA5'] = EMAIndicator(data['Close'], window=5).ema_indicator()
    data['EMA10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
    data['EMA20'] = EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
    data['Signal_EMA_Ribbon'] = np.where(
        (data['EMA5'] > data['EMA10']) & (data['EMA10'] >
                                          data['EMA20']) & (data['EMA20'] > data['EMA50']), 1,
        np.where((data['EMA5'] < data['EMA10']) & (data['EMA10'] < data['EMA20']) & (data['EMA20'] < data['EMA50']), -1, 0))

    # Strategy 7: Volume Price Trend
    data['VPT'] = (data['Volume'] * ((data['Close'] -
                                      data['Close'].shift(1)) / data['Close'].shift(1))).cumsum()
    data['Signal_VPT'] = np.where(
        data['VPT'] > data['VPT'].rolling(window=20).mean(), 1, -1)

    # Strategy 8: Commodity Channel Index (CCI)
    data['CCI'] = (data['Close'] - data['Close'].rolling(window=20).mean()
                   ) / (0.015 * data['Close'].rolling(window=20).std())
    data['Signal_CCI'] = np.where(
        data['CCI'] > 100, -1, np.where(data['CCI'] < -100, 1, 0))

    # Combine signals
    data['Combined_Signal'] = (data['Signal_SMA'] + data['Signal_RSI'] + data['Signal_MACD'] + data['Signal_BB'] +
                               data['Signal_Stoch'] + data['Signal_EMA_Ribbon'] + data['Signal_VPT'] + data['Signal_CCI'])

    return data


def execute_trade(symbol, action, price, amount):
    if action == 'Buy':
        if st.session_state.balance >= amount * price:
            st.session_state.balance -= amount * price
            if symbol in st.session_state.positions:
                total_quantity = st.session_state.positions[symbol]['quantity'] + amount
                total_cost = (st.session_state.positions[symbol]['cost_basis'] *
                              st.session_state.positions[symbol]['quantity']) + (price * amount)
                st.session_state.positions[symbol]['quantity'] = total_quantity
                st.session_state.positions[symbol]['cost_basis'] = total_cost / \
                    total_quantity
            else:
                st.session_state.positions[symbol] = {
                    'quantity': amount, 'cost_basis': price}
            st.session_state.trade_history.append({
                'Date': datetime.now(),
                'Symbol': symbol,
                'Action': action,
                'Price': price,
                'Amount': amount
            })
            st.session_state.stop_loss[symbol] = price * 0.95
            st.session_state.balance_history.append(
                {'Date': datetime.now(), 'Balance': st.session_state.balance})
            return True
    elif action == 'Sell':
        if symbol in st.session_state.positions and st.session_state.positions[symbol]['quantity'] >= amount:
            st.session_state.balance += amount * price
            st.session_state.positions[symbol]['quantity'] -= amount
            if st.session_state.positions[symbol]['quantity'] == 0:
                del st.session_state.positions[symbol]
                if symbol in st.session_state.stop_loss:
                    del st.session_state.stop_loss[symbol]
            st.session_state.trade_history.append({
                'Date': datetime.now(),
                'Symbol': symbol,
                'Action': action,
                'Price': price,
                'Amount': amount
            })
            st.session_state.balance_history.append(
                {'Date': datetime.now(), 'Balance': st.session_state.balance})
            return True
    return False


def check_stop_loss(symbol, current_price):
    if symbol in st.session_state.stop_loss and current_price <= st.session_state.stop_loss[symbol]:
        amount = st.session_state.positions[symbol]['quantity']
        if execute_trade(symbol, 'Sell', current_price, amount):
            st.warning(
                f"Stop-loss triggered for {symbol}. Sold {amount} at ${current_price:.2f}")


def calculate_portfolio_metrics():
    total_value = st.session_state.balance
    for symbol, position in st.session_state.positions.items():
        data = yf.download(
            symbol, start=datetime.now() - timedelta(days=5), end=datetime.now())
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            total_value += position['quantity'] * current_price

    initial_balance = 100000
    roi = (total_value - initial_balance) / initial_balance * 100

    portfolio_values = [initial_balance] + \
        [entry['Balance'] for entry in st.session_state.balance_history[1:]]

    # Corrected max drawdown calculation
    portfolio_series = pd.Series(portfolio_values)
    cumulative_max = portfolio_series.cummax()
    drawdowns = (cumulative_max - portfolio_series) / cumulative_max
    max_drawdown = drawdowns.max() * 100

    return roi, max_drawdown


def calculate_profit_loss(symbol, current_price):
    if symbol in st.session_state.positions:
        position = st.session_state.positions[symbol]
        quantity = position['quantity']
        cost_basis = position['cost_basis']
        profit_loss = (current_price - cost_basis) * quantity
        return profit_loss
    else:
        return 0.0


def plot_balance_history():
    df = pd.DataFrame(st.session_state.balance_history)
    fig = px.line(df, x='Date', y='Balance', title='Account Balance History')
    return fig


def main():
    st.title("Professional Multi-Asset Trading Bot")

    st.sidebar.header("Settings")
    asset_type = st.sidebar.selectbox(
        "Select Asset Type", ["Commodities", "Forex", "Crypto"])

    if asset_type == "Commodities":
        symbols = COMMODITIES
    elif asset_type == "Forex":
        symbols = FOREX_SYMBOLS
    else:
        symbols = CRYPTO_SYMBOLS

    start_date = st.sidebar.date_input(
        "Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    data_source = st.sidebar.radio(
        "Select Data Source", ["yfinance", "Alpha Vantage"])

    trade_amount = st.sidebar.number_input(
        "Trade Amount per Asset", min_value=0.01, value=1.0, step=0.01)

    if st.sidebar.button("Run Analysis"):
        all_data = {}
        signals_list = []
        for symbol in symbols:
            data = get_data(symbol, start_date, end_date, data_source)
            if data.empty:
                continue
            data = calculate_signals(data)
            all_data[symbol] = data
            latest_data = data.iloc[-1]

            combined_signal = latest_data['Combined_Signal']
            current_price = latest_data['Close']

            # Determine recommended action based on combined signal
            if combined_signal >= 3:
                action = 'Buy'
            elif combined_signal <= -3:
                action = 'Sell'
            else:
                action = 'Hold'

            # Execute trade automatically
            if action == 'Buy':
                execute_trade(symbol, 'Buy', current_price, trade_amount)
            elif action == 'Sell':
                # Sell all holdings for the symbol
                if symbol in st.session_state.positions:
                    amount_to_sell = st.session_state.positions[symbol]['quantity']
                    execute_trade(symbol, 'Sell',
                                  current_price, amount_to_sell)

            # Calculate profit/loss
            profit_loss = calculate_profit_loss(symbol, current_price)

            signals_list.append({
                'Symbol': symbol,
                'Current Price': current_price,
                'Combined Signal': combined_signal,
                'Recommended Action': action,
                'Profit/Loss': profit_loss
            })

            # Check stop-loss
            check_stop_loss(symbol, current_price)

        # Display signals in a table
        signals_df = pd.DataFrame(signals_list)
        signals_df = signals_df.sort_values(
            by='Combined Signal', ascending=False)
        st.subheader("Market Signals")
        st.dataframe(signals_df.style.applymap(
            lambda x: 'background-color: lightgreen' if x == 'Buy' else (
                'background-color: pink' if x == 'Sell' else ''),
            subset=['Recommended Action']
        ))

        # Display current positions
        st.subheader("Current Positions")
        positions_list = []
        for symbol, position in st.session_state.positions.items():
            data = all_data.get(symbol)
            if data is not None:
                current_price = data['Close'].iloc[-1]
                quantity = position['quantity']
                cost_basis = position['cost_basis']
                market_value = quantity * current_price
                profit_loss = (current_price - cost_basis) * quantity
                positions_list.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Cost Basis': cost_basis,
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Profit/Loss': profit_loss
                })
        if positions_list:
            positions_df = pd.DataFrame(positions_list)
            st.dataframe(positions_df)
        else:
            st.write("No positions currently held.")

        # Display trade history
        st.subheader("Trade History")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(history_df)
        else:
            st.write("No trades executed yet.")

        # Display account information
        st.subheader("Account Information")
        st.write(f"Current Balance: ${st.session_state.balance:.2f}")
        roi, max_drawdown = calculate_portfolio_metrics()
        st.write(f"ROI: {roi:.2f}%")
        st.write(f"Max Drawdown: {max_drawdown:.2f}%")

        # Display balance history chart
        st.subheader("Account Balance History")
        st.plotly_chart(plot_balance_history())

        # Additional metrics and charts can be added here as needed


if __name__ == "__main__":
    main()
