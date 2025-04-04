import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# --- Streamlit UI ---
st.title("📈 Stock Price Predictor (LSTM)")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert Date column (Fix potential timezone warnings)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Ensure correct column names
    df.columns = df.columns.str.lower()
    
    # Select Features
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in feature_columns):
        st.error("Missing required columns: Open, High, Low, Close, Volume")
        st.stop()
    
    df = df[['date'] + feature_columns]
    df.sort_values('date', inplace=True)

    # --- Normalize Data ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    # --- Prepare LSTM Training Data ---
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

    # --- Build LSTM Model ---
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, len(feature_columns))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(2)  # Predicting Close & Volume
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # --- Make Predictions ---
    predictions = model.predict(X_test)
    
    # Convert Predictions Back to Original Scale
    dummy_features = np.zeros((len(predictions), len(feature_columns)))
    dummy_features[:, [3, 4]] = predictions  # Insert only Close & Volume
    predictions = scaler.inverse_transform(dummy_features)[:, [3, 4]]

    # Convert y_test Back to Original Scale
    dummy_features[:, [3, 4]] = y_test
    actual_prices = scaler.inverse_transform(dummy_features)[:, [3, 4]]

    # --- Future Forecast ---
    def predict_future(days=30):
        future_inputs = X_test[-1]
        future_predictions = []

        for _ in range(days):
            next_pred = model.predict(future_inputs.reshape(1, lookback, len(feature_columns)))
            next_pred_scaled = np.concatenate([future_inputs[1:], next_pred.reshape(1, -1)], axis=0)
            future_inputs = next_pred_scaled
            future_predictions.append(next_pred[0])

        future_predictions = np.array(future_predictions)
        dummy_features = np.zeros((future_predictions.shape[0], len(feature_columns)))
        dummy_features[:, [3, 4]] = future_predictions
        future_predictions = scaler.inverse_transform(dummy_features)[:, [3, 4]]

        return future_predictions

    # --- Streamlit UI: Forecast Window ---
    forecast_days = st.selectbox("📅 Select Forecast Period", [7, 15, 30, 90, 180])
    future_preds = predict_future(forecast_days)

    # Generate Future Dates
    future_dates = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=forecast_days)

    # --- Plot Results ---
    st.subheader("📊 Stock Price Trend (Actual vs Predicted)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual_prices[:, 0], label="Actual Close Price", color='blue')
    ax.plot(predictions[:, 0], label="Predicted Close Price", color='orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # --- Display Forecast Table ---
    predictions_df = pd.DataFrame(future_preds, columns=['Close Price', 'Volume'])
    predictions_df.insert(0, 'Date', future_dates)
    
    st.subheader("📋 Predicted Stock Prices & Volume")
    st.write(predictions_df)
