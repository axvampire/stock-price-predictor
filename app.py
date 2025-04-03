import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Streamlit UI
st.title("📈 Stock Price Predictor (LSTM)")

uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Display Raw Data
    st.subheader("Raw Data")
    st.write(df.head())

    # Select closing prices
    data = df[['Close']].values  # Only use 'Close' price for prediction

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences (lookback = 60 days)
    def create_sequences(data, lookback=60):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

    lookback = 60  # Use past 60 days to predict next day
    X, y = create_sequences(data_scaled, lookback)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.text("Training the LSTM model... (This may take a while)")
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

    # Prepare the actual values for comparison
    actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

    # Plot results
    st.subheader("Stock Price Prediction")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual_prices, label="Actual Prices", color='blue')
    ax.plot(predictions, label="Predicted Prices", color='orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)
