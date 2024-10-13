import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load and preprocess the data
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(url)
df['date'] = pd.to_datetime(df['date'])

# Calculate rolling averages
for col in ['new_cases', 'new_deaths']:
    df[f'{col}_7day_avg'] = df.groupby('location')[col].rolling(
        window=7, min_periods=1).mean().reset_index(0, drop=True)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Added for Heroku deployment

# Rest of your app code remains the same...

# At the end of the file
if __name__ == '__main__':
    app.run_server(debug=False)