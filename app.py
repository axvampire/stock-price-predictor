import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit App Title
st.title("📈 Stock Price Predictor")

# Sidebar File Uploader
st.sidebar.header("Upload Stock Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload your stock data file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Ensure 'Date' is datetime and sort values
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Selecting Features and Target
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict Future Prices
    df['Predicted'] = model.predict(X)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10,5))
    sns.lineplot(x=df['Date'], y=df['Close'], label="Actual")
    sns.lineplot(x=df['Date'], y=df['Predicted'], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    st.pyplot(plt)
