import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Function to load data from CSV file or URL
@st.cache_data
def load_data(file_or_url):
    if isinstance(file_or_url, io.IOBase):
        return pd.read_csv(file_or_url)
    else:
        return pd.read_csv(file_or_url)

# Function to clean data
def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('Unknown')
        else:
            df[column] = df[column].fillna(df[column].mean())
    
    return df

# Main Streamlit app
def main():
    st.title('Data Analysis and Visualization App')

    # File upload or URL input
    data_source = st.radio("Choose data source:", ('Upload CSV', 'Enter CSV URL'))
    
    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
    else:
        url = st.text_input("Enter the URL of the CSV file:")
        if url:
            df = load_data(url)
    
    if 'df' in locals():
        st.write("Data loaded successfully!")
        
        # Data cleaning
        if st.button('Clean Data'):
            df = clean_data(df)
            st.write("Data cleaned!")
        
        # Display raw data
        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(df)
        
        # Display basic statistics
        if st.checkbox('Show basic statistics'):
            st.subheader('Basic Statistics')
            st.write(df.describe())
        
        # Column selection for visualization
        st.subheader('Data Visualization')
        columns = df.columns.tolist()
        x_axis = st.selectbox('Choose the X-axis', options=columns)
        y_axis = st.selectbox('Choose the Y-axis', options=columns)
        
        # Chart type selection
        chart_type = st.radio("Choose the chart type:", ('Scatter', 'Line', 'Bar'))
        
        # Create and display the chart
        if chart_type == 'Scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == 'Line':
            fig = px.line(df, x=x_axis, y=y_axis)
        else:
            fig = px.bar(df, x=x_axis, y=y_axis)
        
        st.plotly_chart(fig)
        
        # Correlation matrix
        if st.checkbox('Show correlation matrix'):
            st.subheader('Correlation Matrix')
            corr = df.corr()
            fig_corr = px.imshow(corr)
            st.plotly_chart(fig_corr)
        
        # Data processing: group by and aggregate
        st.subheader('Data Processing')
        group_by_column = st.selectbox('Group by:', options=columns)
        agg_column = st.selectbox('Aggregate:', options=columns)
        agg_function = st.selectbox('Aggregation function:', options=['mean', 'sum', 'count', 'min', 'max'])
        
        if st.button('Process Data'):
            grouped_df = df.groupby(group_by_column).agg({agg_column: agg_function}).reset_index()
            st.write(grouped_df)
            
            # Visualize processed data
            fig_processed = px.bar(grouped_df, x=group_by_column, y=agg_column)
            st.plotly_chart(fig_processed)

if __name__ == '__main__':
    main()