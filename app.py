import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Load the trained model
MODEL_PATH = "stock_price_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("Model file 'stock_price_model.h5' not found!")
    st.stop()
model = load_model(MODEL_PATH)

# Load stock data
@st.cache_data
def load_data():
    file_path = "stock_data.csv"
    if not os.path.exists(file_path):
        st.error("Error: 'stock_data.csv' not found!")
        return None
    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        st.error("Error: Missing 'Date' column in CSV file.")
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

# Prepare Data for Prediction
def prepare_data(df, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Close"]].values)
    
    X = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

# Predict Future Prices
def predict_future(df, days=30):
    look_back = 30
    X, scaler = prepare_data(df, look_back)
    
    last_sequence = X[-1]  # Last available input sequence
    future_predictions = []
    
    for _ in range(days):
        pred = model.predict(last_sequence.reshape(1, look_back, 1))
        future_predictions.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = pred
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    
    return pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})

# Streamlit UI
st.title("Stock Price Prediction (2024 & Beyond)")

# Load Data
stock_data = load_data()
if stock_data is not None:
    st.write("### Recent Stock Data (End of 2024)")
    st.line_chart(stock_data["Close"].tail(100))
    
    # Predict Future Prices
    st.write("### Future Predictions")
    future_df = predict_future(stock_data, days=30)
    st.write(future_df)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data.index[-100:], stock_data["Close"].tail(100), label="Actual Prices")
    ax.plot(future_df["Date"], future_df["Predicted Price"], label="Predicted Prices", linestyle="dashed", color='red')
    ax.set_title("Stock Prices: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
