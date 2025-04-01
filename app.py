import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to Load and Clean Data
@st.cache_data
def load_data():
    file_path = "stock_data.csv"

    try:
        # Load CSV
        df = pd.read_csv(file_path)

        # Ensure 'Date' exists
        if "Date" not in df.columns:
            st.error("Error: Missing 'Date' column in CSV file.")
            return None

        # Handle NaN values in 'Date'
        df = df.dropna(subset=["Date"])

        # Ensure 'Date' column is string before conversion
        df["Date"] = df["Date"].astype(str)

        # Convert 'Date' column to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Drop any remaining NaT (invalid date) values
        df = df.dropna(subset=["Date"])

        # Set Date as index
        df.set_index("Date", inplace=True)

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Read stock data
stock_data = load_data()

# Show the first few rows of the dataset
st.write("### Preview of Stock Data", stock_data.head())

# Plot stock prices
st.write("### Stock Price Chart")
fig, ax = plt.subplots()
ax.plot(stock_data.index, stock_data["Close"], label="Close Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# ---- MACHINE LEARNING MODEL FOR PREDICTION ----

# Prepare Data for Prediction
stock_data["Days"] = np.arange(len(stock_data))  # Convert dates to numerical values
X = stock_data[["Days"]]
y = stock_data["Close"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Fix: train_test_split is now defined

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Future Prices
future_days = 30  # Predict 30 days ahead
future_X = np.arange(len(stock_data), len(stock_data) + future_days).reshape(-1, 1)
future_predictions = model.predict(future_X)

# Plot Predictions
st.write("### Stock Price Prediction for Next 30 Days")
fig2, ax2 = plt.subplots()
ax2.plot(stock_data.index, stock_data["Close"], label="Actual Prices", color="blue")
ax2.plot(pd.date_range(start=stock_data.index[-1], periods=future_days, freq='D'), 
         future_predictions, label="Predicted Prices", color="red", linestyle="dashed")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)  #  Fix: This ensures the prediction chart is displayed in Streamlit
