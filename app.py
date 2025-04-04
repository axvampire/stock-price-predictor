import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Streamlit UI
st.title("📈 Stock Price Predictor (LSTM)")

# User configurable hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
lstm_units = st.sidebar.slider("LSTM Units", min_value=10, max_value=200, step=10, value=50)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.05, value=0.2)
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, step=5, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)

# Upload file
uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert Date column (Fix potential timezone warnings)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Ensure correct column names
    df.columns = df.columns.str.lower()
    
    # Select features
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in feature_columns):
        st.error("Missing required columns: Open, High, Low, Close, Volume")
        st.stop()
    
    df = df[['date'] + feature_columns]
    df.sort_values('date', inplace=True)

    # Show Raw Dataset (Last 10 Rows)
    st.subheader("📊 Raw Data Preview")
    st.write(df.tail(10))

    # Normalize Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    # Prepare LSTM Training Data
    lookback = 60  # Days of history for LSTM
    X, y = [], []

    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, [3, 4]])  # Predicting Close & Volume

    X, y = np.array(X), np.array(y)

    # Split Data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM Model
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(lookback, len(feature_columns))),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(2)  # Predicting Close & Volume
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make Predictions
    predictions = model.predict(X_test)
    
    # Convert predictions back to original scale
    dummy_features = np.zeros((len(predictions), len(feature_columns)))
    dummy_features[:, [3, 4]] = predictions  # Insert only Close & Volume
    predictions = scaler.inverse_transform(dummy_features)[:, [3, 4]]

    # Convert y_test back to original scale
    dummy_features[:, [3, 4]] = y_test
    actual_prices = scaler.inverse_transform(dummy_features)[:, [3, 4]]

    # Future Forecast
    def predict_future(days=30):
        future_inputs = X_test[-1]  # Start with the last known data
        future_predictions = []

        for _ in range(days):
            next_pred = model.predict(future_inputs.reshape(1, lookback, len(feature_columns)))

            # Ensure predicted values are mapped to the correct indices (Close & Volume)
            next_pred_filled = np.zeros((1, len(feature_columns)))  # Create a row with 5 columns
            next_pred_filled[:, [3, 4]] = next_pred  # Insert predictions into Close & Volume columns

            # Shift input for next prediction
            future_inputs = np.vstack([future_inputs[1:], next_pred_filled])  # Stack vertically
            future_predictions.append(next_pred[0])

        future_predictions = np.array(future_predictions)

        # Convert predictions back to original scale
        dummy_features = np.zeros((future_predictions.shape[0], len(feature_columns)))
        dummy_features[:, [3, 4]] = future_predictions  # Insert Close & Volume
        future_predictions = scaler.inverse_transform(dummy_features)[:, [3, 4]]

        return future_predictions

    # Streamlit UI: Forecast Window
    forecast_days = st.selectbox("📅 Select Forecast Period", [7, 15, 30, 90, 180])
    future_preds = predict_future(forecast_days)

    # Generate future dates
    future_dates = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=forecast_days)

    # Plot results
    st.subheader("📊 Stock Price Trend (Actual vs Predicted)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual_prices[:, 0], label="Actual Close Price", color='blue')
    ax.plot(predictions[:, 0], label="Predicted Close Price", color='orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    #Display forecast table
    predictions_df = pd.DataFrame(future_preds, columns=['Close Price', 'Volume'])
    predictions_df.insert(0, 'Date', future_dates)
    
    st.subheader("📋 Predicted Stock Prices & Volume")
    st.write(predictions_df)
