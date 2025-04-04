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

# --- UI for Hyperparameter Tuning ---
st.sidebar.header("🔧 Model Settings")
lstm_units = st.sidebar.slider("LSTM Units", min_value=32, max_value=256, value=100, step=16)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=20, step=5)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)

forecast_days = st.sidebar.radio("📅 Forecast Window", [7, 15, 30, 90, 180], index=2)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
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
        LSTM(lstm_units, return_sequences=True, input_shape=(lookback, len(feature_columns))),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.text("Training the optimized LSTM model... (This may take a few minutes)")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

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

    # --- Future Prediction ---
    st.subheader(f"Future Stock Price Prediction for {forecast_days} Days")

    def predict_future(days=30):
        future_inputs = X_test[-1]  # Last known sequence
        future_predictions = []

        for _ in range(days):
            next_pred = model.predict(future_inputs.reshape(1, lookback, len(feature_columns)))
            next_pred_scaled = np.concatenate([future_inputs[1:], next_pred.reshape(1, -1)], axis=0)
            future_inputs = next_pred_scaled
            future_predictions.append(next_pred[0,0])

        # Convert predictions back to actual values
        future_predictions = scaler.inverse_transform(
            np.concatenate([np.tile(X_test[-1, -1, :-1], (days, 1)), np.array(future_predictions).reshape(-1,1)], axis=1)
        )[:, 3]

        return future_predictions

    future_preds = predict_future(forecast_days)

    # Plot future predictions
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(len(future_preds)), future_preds, label=f"Next {forecast_days} Days", color='green')
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Stock Price")
    ax.legend()
    st.pyplot(fig)
