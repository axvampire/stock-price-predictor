import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from fbprophet import Prophet

# Load Data
st.title("📈 Stock Price Predictor")
st.sidebar.header("Upload Stock Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Preprocessing (Assuming Date column exists)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Train Prophet Model
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot results
    fig = px.line(forecast, x='ds', y='yhat', title='Stock Price Prediction')
    st.plotly_chart(fig)
