import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io

# Set page configuration
st.set_page_config(page_title="Comprehensive Data Analysis Application", layout="wide")

# Initialize session state for DataFrame
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

def main():
    st.title("Comprehensive Data Analysis Application")

    # Sidebar navigation
    menu = ["Data Upload", "Data Cleansing", "Data Visualization", "Data Analysis", "Advanced Analytics"]
    choice = st.sidebar.selectbox("Select Action", menu)

    if choice == "Data Upload":
        data_upload()
    elif choice == "Data Cleansing":
        data_cleansing()
    elif choice == "Data Visualization":
        data_visualization()
    elif choice == "Data Analysis":
        data_analysis()
    elif choice == "Advanced Analytics":
        advanced_analytics()

def data_upload():
    st.subheader("Upload Your Data")

    data_source = st.radio("Choose data source:", ('Upload CSV', 'Enter CSV URL'))

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success("Data uploaded successfully!")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.info("Awaiting for CSV file to be uploaded.")
    else:
        url = st.text_input("Enter the URL of the CSV file:")
        if url:
            try:
                df = pd.read_csv(url)
                st.session_state['df'] = df
                st.success("Data loaded successfully from URL!")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading data from URL: {e}")

def data_cleansing():
    st.subheader("Data Cleansing")

    if not st.session_state['df'].empty:
        df = st.session_state['df'].copy()
        st.write("Data Preview:", df.head())

        with st.expander("Handle Missing Values"):
            missing_value_option = st.selectbox("Choose how to handle missing values:", 
                                                ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"])

            if st.button("Apply Missing Value Treatment"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                if missing_value_option == "Fill with Mean":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif missing_value_option == "Fill with Median":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif missing_value_option == "Fill with Mode":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
                    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
                elif missing_value_option == "Drop Rows":
                    df = df.dropna()
                st.session_state['df'] = df
                st.success("Missing values handled successfully!")
                st.write(df.head())

        with st.expander("Remove Duplicates"):
            if st.button("Remove Duplicate Rows"):
                before_rows = df.shape[0]
                df = df.drop_duplicates()
                after_rows = df.shape[0]
                st.session_state['df'] = df
                st.success(f"Duplicates removed! {before_rows - after_rows} rows dropped.")
                st.write(df.head())

        with st.expander("Data Type Conversion"):
            st.write("Convert Data Types of Columns")
            conversion_col = st.selectbox("Select Column to Convert", df.columns)
            conversion_type = st.selectbox("Convert to Type", ["Integer", "Float", "String", "DateTime"])

            if st.button("Convert Data Type"):
                try:
                    if conversion_type == "Integer":
                        df[conversion_col] = df[conversion_col].astype(int)
                    elif conversion_type == "Float":
                        df[conversion_col] = df[conversion_col].astype(float)
                    elif conversion_type == "String":
                        df[conversion_col] = df[conversion_col].astype(str)
                    elif conversion_type == "DateTime":
                        df[conversion_col] = pd.to_datetime(df[conversion_col])
                    st.session_state['df'] = df
                    st.success(f"Column '{conversion_col}' converted to {conversion_type}")
                    st.write(df.head())
                except Exception as e:
                    st.error(f"Error converting column: {e}")

        with st.expander("Outlier Detection and Handling"):
            st.write("Detect and Handle Outliers")
            outlier_col = st.selectbox("Select Numeric Column for Outlier Detection", df.select_dtypes(include=np.number).columns)
            method = st.selectbox("Choose Outlier Handling Method", ["Remove Outliers", "Cap at Percentiles"])

            if st.button("Handle Outliers"):
                if method == "Remove Outliers":
                    q1 = df[outlier_col].quantile(0.25)
                    q3 = df[outlier_col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    before_rows = df.shape[0]
                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                    after_rows = df.shape[0]
                    st.session_state['df'] = df
                    st.success(f"Outliers removed! {before_rows - after_rows} rows dropped.")
                    st.write(df.head())
                elif method == "Cap at Percentiles":
                    lower_percentile = df[outlier_col].quantile(0.05)
                    upper_percentile = df[outlier_col].quantile(0.95)
                    df[outlier_col] = np.clip(df[outlier_col], lower_percentile, upper_percentile)
                    st.session_state['df'] = df
                    st.success(f"Outliers capped at 5th and 95th percentiles.")
                    st.write(df.head())
    else:
        st.warning("Please upload data first.")

def data_visualization():
    st.subheader("Data Visualization")

    if not st.session_state['df'].empty:
        df = st.session_state['df']
        st.write("Data Preview:", df.head())

        plot_types = ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"]
        plot_type = st.selectbox("Select Plot Type", plot_types)

        if plot_type != "Heatmap":
            columns = df.columns.tolist()
            x_axis = st.selectbox('Select X-axis', options=columns)
            y_axis = st.selectbox('Select Y-axis', options=columns)
            color_option = st.selectbox('Select Color Grouping', options=[None] + columns)

        if st.button("Generate Plot"):
            try:
                if plot_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option)
                elif plot_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_option)
                elif plot_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_option)
                elif plot_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_option)
                elif plot_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_option)
                elif plot_type == "Heatmap":
                    corr = df.corr()
                    fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating plot: {e}")
    else:
        st.warning("Please upload and cleanse data first.")

def data_analysis():
    st.subheader("Data Analysis")

    if not st.session_state['df'].empty:
        df = st.session_state['df']
        st.write("Data Preview:", df.head())

        with st.expander("Descriptive Statistics"):
            if st.checkbox('Show Descriptive Statistics'):
                st.write(df.describe())

        with st.expander("Correlation Matrix"):
            if st.checkbox('Show Correlation Matrix'):
                corr = df.corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig)

        with st.expander("Grouping and Aggregation"):
            st.write("### Grouping and Aggregation")
            columns = df.columns.tolist()
            group_by_column = st.selectbox('Group by:', options=columns)
            agg_column = st.selectbox('Aggregate:', options=columns)
            agg_function = st.selectbox('Aggregation Function:', ['mean', 'sum', 'count', 'min', 'max'])

            if st.button('Perform Grouping and Aggregation'):
                try:
                    grouped_df = df.groupby(group_by_column).agg({agg_column: agg_function}).reset_index()
                    st.write(grouped_df)
                    fig = px.bar(grouped_df, x=group_by_column, y=agg_column)
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error in grouping and aggregation: {e}")
    else:
        st.warning("Please upload and cleanse data first.")

def advanced_analytics():
    st.subheader("Advanced Analytics")

    if not st.session_state['df'].empty:
        df = st.session_state['df']
        st.write("Data Preview:", df.head())

        analytics_choice = st.selectbox("Select Advanced Analytics Technique", ["Time Series Forecasting", "Clustering"])

        if analytics_choice == "Time Series Forecasting":
            time_series_forecasting(df)
        elif analytics_choice == "Clustering":
            clustering_analysis(df)
    else:
        st.warning("Please upload and cleanse data first.")

def time_series_forecasting(df):
    st.write("### Time Series Forecasting")
    date_columns = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not date_columns:
        st.error("No date columns found in the dataset. Please make sure you have a date column.")
        return

    date_column = st.selectbox("Select Date Column", date_columns)
    target_column = st.selectbox("Select Target Column for Forecasting", numeric_columns)

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df_forecast = df[[date_column, target_column]].dropna()
    df_forecast = df_forecast.rename(columns={date_column: 'ds', target_column: 'y'})

    periods_input = st.number_input('Select number of periods for forecasting:', min_value=1, max_value=365, value=30)

    if st.button("Perform Forecasting"):
        try:
            model = Prophet()
            model.fit(df_forecast)
            future = model.make_future_dataframe(periods=periods_input)
            forecast = model.predict(future)
            st.write("Forecasted Data:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1, use_container_width=True)

            # Show components
            st.write("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.write(fig2)
        except Exception as e:
            st.error(f"Error in forecasting: {e}")

def clustering_analysis(df):
    st.write("### K-Means Clustering")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        st.error("Need at least two numeric columns for clustering.")
        return

    selected_columns = st.multiselect("Select Columns for Clustering", numeric_columns, default=numeric_columns)
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

    if st.button("Perform Clustering"):
        try:
            data = df[selected_columns].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            df['Cluster'] = clusters
            st.write("Clustered Data Preview:")
            st.write(df.head())

            # Visualize clusters (only if 2 selected columns)
            if len(selected_columns) == 2:
                fig_cluster = px.scatter(
                    df,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    color='Cluster',
                    title="Clustering Visualization"
                )
                st.plotly_chart(fig_cluster)
            else:
                st.warning("Visualization is available when exactly 2 columns are selected.")
        except Exception as e:
            st.error(f"Error in clustering: {e}")

if __name__ == '__main__':
    main()