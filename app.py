import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


st.title("📈 Stock Price Predictor")
ticker = st.text_input("Enter stock symbol (e.g., AAPL):")

if st.button("Predict"):
    stock_data = yf.download(ticker, period="5y")
    st.line_chart(stock_data["Close"])


# Train Model
def train_model(df):
    df = df[['Close']]
    df['Target'] = df['Close'].shift(-30)
    df.dropna(inplace=True)
    
    X = np.array(df.drop(columns=['Target']))
    y = np.array(df['Target'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict Future Prices
def predict_future(df, model, days=30):
    last_data = np.array(df[['Close']].tail(1))
    future_predictions = []
    future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    
    for _ in range(days):
        pred = model.predict(last_data.reshape(1, -1))[0]
        future_predictions.append(pred)
        last_data = np.array([[pred]])
    
    return pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})

# Streamlit UI
st.title("Stock Price Prediction (2024 & Beyond)")

# Load Data
stock_data = load_data()
if stock_data is not None:
    st.write("### Recent Stock Data (End of 2024)")
    st.line_chart(stock_data["Close"].tail(100))
    
    # Train Model
    model = train_model(stock_data)
    
    # Predict Future Prices
    st.write("### Future Predictions")
    future_df = predict_future(stock_data, model, days=30)
    st.write(future_df)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data.index[-100:], stock_data["Close"].tail(100), label="Actual Prices")
    ax.plot(future_df["Date"], future_df["Predicted Price"], label="Predicted Prices", linestyle="dashed", color='red')
    ax.set_title("Stock Prices: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
