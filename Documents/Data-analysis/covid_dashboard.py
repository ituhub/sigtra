import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import io
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cmdstanpy
from cmdstanpy import install_cmdstan

# Set page config
st.set_page_config(page_title="Advanced COVID-19 Dashboard", layout="wide")

@st.cache_data(ttl=3600)
def load_data(url='https://covid.ourworldindata.org/data/owid-covid-data.csv'):
    """
    Loads COVID-19 data from Our World in Data.

    Returns:
        df (pd.DataFrame): The loaded DataFrame with a datetime 'date' column.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def process_data(df):
    """
    Processes the loaded data by handling missing values, calculating rolling averages,
    growth rates, and additional metrics.

    Args:
        df (pd.DataFrame): The loaded DataFrame.

    Returns:
        df (pd.DataFrame): The processed DataFrame.
    """
    if df.empty:
        return df

    # Handle missing data
    for col in ['new_cases', 'new_deaths', 'new_vaccinations']:
        df[col] = df[col].fillna(0)

    # Calculate rolling averages and growth rates
    for col in ['new_cases', 'new_deaths', 'new_vaccinations']:
        df[f'{col}_7day_avg'] = df.groupby('location')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df[f'{col}_growth_rate'] = df.groupby('location')[f'{col}_7day_avg'].pct_change(periods=7)
        df[f'{col}_growth_rate'] = df[f'{col}_growth_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Avoid division by zero in 'population'
    df['population'] = df['population'].replace(0, np.nan)

    # Create additional metrics
    df['cases_per_million'] = (df['total_cases'] * 1e6 / df['population']).fillna(0)
    df['deaths_per_million'] = (df['total_deaths'] * 1e6 / df['population']).fillna(0)

    return df

# Load and process data
df = load_data()
df = process_data(df)

# Sidebar for user input
st.sidebar.header('Dashboard Controls')

if not df.empty:
    available_countries = sorted(df['location'].unique())
    countries = st.sidebar.multiselect('Select Countries', options=available_countries,
                                       default=['United States', 'India', 'Brazil'])
    metrics = ['total_cases', 'new_cases_7day_avg', 'total_deaths', 'new_deaths_7day_avg',
               'total_vaccinations', 'new_vaccinations_7day_avg', 'cases_per_million', 'deaths_per_million']
    metric = st.sidebar.selectbox('Select Primary Metric', metrics)
    date_range = st.sidebar.date_input('Select Date Range',
                                       [df['date'].min(), df['date'].max()],
                                       min_value=df['date'].min(),
                                       max_value=df['date'].max())
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]

    # Filter data based on user input
    filtered_df = df[(df['location'].isin(countries)) &
                     (df['date'] >= pd.to_datetime(start_date)) &
                     (df['date'] <= pd.to_datetime(end_date))]
else:
    st.error("No data available. Please check your internet connection and try again.")
    st.stop()

# Main dashboard
st.title('Advanced COVID-19 Data Analysis Dashboard')

# Global Overview Section
st.header('Global Overview')

# Filter latest global data and handle missing iso_codes
available_dates = df[df[metric].notna()]['date']
if not available_dates.empty:
    most_recent_date = available_dates.max()
    latest_global_data = df[df['date'] == most_recent_date]

    # Remove rows with missing or invalid iso_codes
    latest_global_data = latest_global_data[latest_global_data['iso_code'].notna()]
    latest_global_data = latest_global_data[~latest_global_data['iso_code'].str.startswith('OWID')]
    latest_global_data = latest_global_data[latest_global_data[metric].notna()]

    if not latest_global_data.empty and metric in latest_global_data.columns and latest_global_data[metric].notna().any():
        fig_map = px.choropleth(
            latest_global_data,
            locations="iso_code",
            locationmode='ISO-3',
            color=metric,
            hover_name="location",
            color_continuous_scale="Viridis",
            range_color=(0, latest_global_data[metric].quantile(0.95)),
            title=f"Global {metric.replace('_', ' ').title()} as of {most_recent_date.strftime('%Y-%m-%d')}"
        )
        fig_map.update_geos(showcountries=True, countrycolor="lightgray")
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)

        # Global statistics
        total_cases = latest_global_data['total_cases'].fillna(0).sum()
        total_deaths = latest_global_data['total_deaths'].fillna(0).sum()
        total_vaccinations = latest_global_data['total_vaccinations'].fillna(0).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Global Total Cases", f"{total_cases:,.0f}")
        col2.metric("Global Total Deaths", f"{total_deaths:,.0f}")
        col3.metric("Global Total Vaccinations", f"{total_vaccinations:,.0f}")
    else:
        st.warning("Insufficient data to display the world map or global statistics.")
else:
    st.warning(f"No available data for metric '{metric}'.")

# Detailed Country Analysis
st.header('Detailed Country Analysis')

if not filtered_df.empty and countries:
    # Multi-metric comparison
    st.subheader('Multi-Metric Comparison')
    latest_country_data = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
    latest_country_data = latest_country_data[latest_country_data['iso_code'].notna()]
    latest_country_data = latest_country_data[~latest_country_data['iso_code'].str.startswith('OWID')]

    if not latest_country_data.empty:
        fig_multi = go.Figure()
        for m in metrics:
            if m in latest_country_data.columns and latest_country_data[m].notna().any():
                fig_multi.add_trace(go.Bar(x=latest_country_data['location'], y=latest_country_data[m],
                                           name=m.replace('_', ' ').title()))

        fig_multi.update_layout(barmode='group', height=500,
                                title='Comparison of Multiple Metrics Across Selected Countries')
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.warning("No data available for the selected countries on the latest date.")

    # Time series analysis
    st.subheader('Time Series Analysis')
    if metric in filtered_df.columns and filtered_df[metric].notna().any():
        fig_time = px.line(filtered_df, x='date', y=metric, color='location',
                           title=f"{metric.replace('_', ' ').title()} Over Time")
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning("Insufficient data to display time series.")

    # Vaccination progress
    st.subheader('Vaccination Progress')
    if 'people_vaccinated_per_hundred' in latest_country_data.columns and latest_country_data['people_vaccinated_per_hundred'].notna().any():
        fig_vax = px.bar(latest_country_data, x='location', y='people_vaccinated_per_hundred',
                         title="Vaccination Progress (% of Population)",
                         labels={'people_vaccinated_per_hundred': 'Vaccinated per Hundred'})
        st.plotly_chart(fig_vax, use_container_width=True)
    else:
        st.warning("Vaccination data not available.")

    # Country-specific details
    st.subheader('Country-Specific Details')
    for country in countries:
        st.write(f"### {country}")
        country_data = latest_country_data[latest_country_data['location'] == country]
        if not country_data.empty:
            col1, col2 = st.columns(2)

            # Display key metrics
            key_metrics = ['total_cases', 'total_deaths', 'total_vaccinations', 'population']
            for i, m in enumerate(key_metrics):
                if m in country_data.columns and pd.notnull(country_data[m].iloc[0]):
                    (col1 if i % 2 == 0 else col2).metric(
                        m.replace('_', ' ').title(),
                        f"{country_data[m].iloc[0]:,.0f}"
                    )

            # Display additional information
            additional_info = ['gdp_per_capita', 'life_expectancy', 'human_development_index']
            for m in additional_info:
                if m in country_data.columns:
                    value = country_data[m].iloc[0]
                    if pd.notnull(value):
                        st.write(f"{m.replace('_', ' ').title()}: {value:.2f}")
        else:
            st.write(f"No data available for {country}")
        st.write("---")
else:
    st.warning("Please select at least one country to view detailed information.")

# Data Insights Section
st.header('Data Insights')

if not filtered_df.empty and metric in filtered_df.columns:
    for country in countries:
        country_data = filtered_df[filtered_df['location'] == country]
        if not country_data.empty and country_data[metric].notna().any():
            st.subheader(f"Insights for {country}")

            latest_value = country_data[metric].iloc[-1]
            peak_value = country_data[metric].max()
            peak_date = country_data.loc[country_data[metric].idxmax(), 'date']

            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Value", f"{latest_value:,.0f}")
            col2.metric("Peak Value", f"{peak_value:,.0f}")
            col3.metric("Peak Date", peak_date.strftime('%Y-%m-%d'))

            # Calculate and display growth rates
            if f"{metric}_growth_rate" in country_data.columns:
                recent_growth = country_data[f"{metric}_growth_rate"].iloc[-1]
                avg_growth = country_data[f"{metric}_growth_rate"].mean()
                st.write(f"Recent 7-day growth rate: {recent_growth:.2%}")
                st.write(f"Average growth rate: {avg_growth:.2%}")

            # Display a mini chart
            if country_data[metric].notna().any():
                fig_mini = px.line(country_data, x='date', y=metric,
                                   title=f"{metric.replace('_', ' ').title()} Trend for {country}")
                st.plotly_chart(fig_mini, use_container_width=True)
            else:
                st.warning(f"No data available for {metric} in {country}.")

            st.write("---")
        else:
            st.warning(f"No data available for {country} or metric '{metric}'.")
else:
    st.warning("Insufficient data to display insights.")

# Advanced Analytics Section
st.header('Advanced Analytics')

if not filtered_df.empty and len(countries) > 0:
    # Time Series Forecasting using Prophet
    st.subheader('Time Series Forecasting')

    forecast_country = st.selectbox('Select a country for forecasting', countries)
    forecast_periods = st.slider('Select number of days to forecast', min_value=7, max_value=60, value=30)

    country_data = df[df['location'] == forecast_country][['date', 'new_cases']]
    country_data = country_data.rename(columns={'date': 'ds', 'new_cases': 'y'})
    country_data = country_data.dropna()

    if not country_data.empty:
        try:
            m = Prophet()
            m.fit(country_data)
            future = m.make_future_dataframe(periods=forecast_periods)
            forecast = m.predict(future)

            # Plot the forecast using plotly
            fig_forecast = plot_plotly(m, forecast)
            fig_forecast.update_layout(title=f"Forecasted New Cases for {forecast_country} (Next {forecast_periods} days)")
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.write("This forecast uses the Prophet model to predict future new cases.")
        except Exception as e:
            st.warning(f"An error occurred while forecasting: {e}")
    else:
        st.warning(f"No data available to perform forecasting for {forecast_country}.")

    # Clustering Countries
    st.subheader('Clustering Countries Based on Pandemic Patterns')

    # Prepare data for clustering
    latest_global_data = df[df['date'] == df['date'].max()]
    latest_global_data = latest_global_data[latest_global_data['iso_code'].notna()]
    latest_global_data = latest_global_data[~latest_global_data['iso_code'].str.startswith('OWID')]

    cluster_features = ['total_cases_per_million', 'total_deaths_per_million',
                        'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million']

    # Ensure the features exist in the data
    for feature in cluster_features:
        if feature not in latest_global_data.columns:
            base_feature = feature.replace('_per_million', '')
            if base_feature in latest_global_data.columns:
                latest_global_data[feature] = (latest_global_data[base_feature] * 1e6 / latest_global_data['population']).fillna(0)
            else:
                latest_global_data[feature] = 0

    cluster_data = latest_global_data[['location'] + cluster_features]
    cluster_data = cluster_data.dropna()
    cluster_data = cluster_data.set_index('location')

    if not cluster_data.empty:
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            kmeans = KMeans(n_clusters=4, random_state=0)
            clusters = kmeans.fit_predict(scaled_data)
            cluster_data['Cluster'] = clusters

            cluster_data.reset_index(inplace=True)

            fig_cluster = px.scatter(cluster_data, x='total_cases_per_million', y='total_deaths_per_million',
                                     color='Cluster', hover_data=['location'],
                                     title='Clustering Countries Based on COVID-19 Impact',
                                     labels={
                                         'total_cases_per_million': 'Total Cases per Million',
                                         'total_deaths_per_million': 'Total Deaths per Million'
                                     })

            st.plotly_chart(fig_cluster, use_container_width=True)

            st.write("Countries are clustered based on cases and deaths per million.")
        except Exception as e:
            st.warning(f"An error occurred during clustering: {e}")
    else:
        st.warning("Insufficient data to perform clustering.")
else:
    st.warning("Please select at least one country for advanced analytics.")

# Footer
st.markdown("---")
st.markdown("Data Source: [Our World in Data](https://ourworldindata.org/coronavirus-source-data)")
st.markdown("Dashboard created with Streamlit and Plotly")
st.markdown("Last updated: " + df['date'].max().strftime('%Y-%m-%d') + " UTC")
