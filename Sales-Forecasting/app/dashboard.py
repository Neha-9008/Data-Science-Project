
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path settings
DATA_PATH = '../data/train.csv'

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df.set_index('Order Date', inplace=True)
    monthly_sales = df['Sales'].resample('MS').sum()
    monthly_sales = monthly_sales.interpolate(method='linear')
    return monthly_sales

def fit_arima_model(data):
    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit()
    return model_fit

def forecast_sales(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, mse, rmse, mape

def main():
    st.title('Sales Forecasting Dashboard')
    st.sidebar.title('Sales Forecasting Options')

    # File Upload
    uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv'])

    if uploaded_file:
        # Load data
        data = load_and_preprocess_data(uploaded_file)

        # Display data preview
        st.write('Data Preview:')
        st.write(data.tail())

        # Plot historical data
        st.subheader('Historical Sales Data')
        fig = px.line(data, x=data.index, y='Sales', title='Monthly Sales')
        st.plotly_chart(fig)

        # Forecast Horizon
        forecast_horizon = st.sidebar.slider('Forecast Horizon (Months)', min_value=1, max_value=12, value=6)

        # Model Training and Forecasting
        st.subheader('ARIMA Forecasting')
        model = fit_arima_model(data)
        forecast = forecast_sales(model, steps=forecast_horizon)

        # Display Forecast
        forecast_index = pd.date_range(start=data.index[-1], periods=forecast_horizon + 1, freq='MS')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)

        st.write(f'Forecast for next {forecast_horizon} months:')
        st.write(forecast_series)

        # Plot Forecast
        fig_forecast = px.line(x=forecast_series.index, y=forecast_series.values, title='Forecasted Sales', labels={'x': 'Date', 'y': 'Sales'})
        st.plotly_chart(fig_forecast)

if __name__ == '__main__':
    main()
