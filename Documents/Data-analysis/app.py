import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet  # Updated import statement

# Streamlit title and description
st.title("AI-Powered Sales Forecasting")
st.write("This app predicts future sales using the Prophet model.")

# Load the dataset
data = pd.read_csv('data/sales_data.csv')

# Display the raw data
st.subheader('Raw Data')
st.write(data.tail())

# Prepare data for Prophet
df_prophet = data[['date', 'sales']].rename(
    columns={'date': 'ds', 'sales': 'y'})

# Plot the sales data
st.subheader('Sales Over Time')
st.line_chart(df_prophet.set_index('ds')['y'])

# Prophet Model
model = Prophet()
model.fit(df_prophet)

# Predict the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Display forecasted data
st.subheader('Forecasted Sales')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast using Plotly
st.subheader('Sales Forecast')
fig1 = model.plot(forecast)
st.plotly_chart(fig1)