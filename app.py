import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Streamlit UI
st.title("📈 Stock Price Predictor (LSTM)")

uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=['csv'])

# UI for Hyperparameter Tuning
st.sidebar.header("🔧 Model Settings")
lstm_units = st.sidebar.slider("LSTM Units", min_value=32, max_value=256, value=64, step=16)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=10, step=5)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=64, step=8)

forecast_days = st.sidebar.radio("📅 Forecast Window", [7, 15, 30, 90, 180], index=2)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Fix applied
    df.sort_values('Date', inplace=True)


    # Display Raw Data
    st.subheader("Raw Data")
    st.write(df.tail())

    # Select important features
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[feature_columns].values  # Using multiple features

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    def create_sequences(data, lookback=60):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback, 3])  # Predicting Close price (index 3)
        return np.array(X), np.array(y)

    lookback = 60
    X, y = create_sequences(data_scaled, lookback)

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build an optimized LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, len(feature_columns))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(2)  # Predicts Close & Volume
    ])


    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.text("Training the optimized LSTM model... (This may take a few minutes)")
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :], predictions.reshape(-1,1)], axis=1)
    )[:, 3]  # Get only the Close price

    # Prepare actual values for comparison
    actual_prices = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :], y_test.reshape(-1,1)], axis=1)
    )[:, 3]

    # Plot historical vs predicted prices
    st.subheader("Stock Price Prediction")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual_prices, label="Actual Prices", color='blue')
    ax.plot(predictions, label="Predicted Prices", color='orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Future Prediction
st.subheader(f"Future Stock Price Prediction for {forecast_days} Days")

def predict_future(days=30):
    future_inputs = X_test[-1]  # Last known sequence
    future_predictions = []

    for _ in range(days):
        next_pred = model.predict(future_inputs.reshape(1, lookback, len(feature_columns)))
        next_pred_scaled = np.concatenate([future_inputs[1:], next_pred.reshape(1, -1)], axis=0)
        future_inputs = next_pred_scaled
        future_predictions.append(next_pred[0])  # Predicts both Close & Volume

    # Convert predictions back to original scale
    future_predictions = np.array(future_predictions)

    # Ensure correct shape before inverse transformation
    dummy_features = np.zeros((future_predictions.shape[0], len(feature_columns)))  # Create placeholder features
    dummy_features[:, [3, 4]] = future_predictions  # Only insert Close & Volume
    future_predictions = scaler.inverse_transform(dummy_features)[:, [3, 4]]  # Extract Close & Volume after inverse transform

    return future_predictions

future_preds = predict_future(forecast_days)

# Generate Future Dates
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:]

# Create DataFrame for Display
predictions_df = pd.DataFrame(future_preds, columns=['Close Price', 'Volume'])
predictions_df.insert(0, 'Date', future_dates)

# Show Graph
st.subheader("📊 Stock Price Trend (Actual vs Predicted)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(actual_prices, label="Actual Prices", color='blue')
ax.plot(predictions, label="Predicted Prices", color='orange')
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

# Show Table Below Graph
st.subheader("📋 Predicted Stock Prices & Volume")
st.write(predictions_df)
