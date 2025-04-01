import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class StockDataProcessor:
    """Class to handle stock data preprocessing and column normalization."""

    def __init__(self, df):
        self.df = df

    def standardize_columns(self):
        """Standardizes column names to lowercase for consistency."""
        self.df.columns = [col.lower() for col in self.df.columns]

    def process_date_column(self):
        """Finds and processes the date column, converting it to datetime format."""
        possible_date_columns = ['date', 'Date', 'DATE']
        for col in possible_date_columns:
            if col.lower() in self.df.columns:
                self.df.rename(columns={col.lower(): 'Date'}, inplace=True)
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                return
        raise ValueError("No recognizable 'Date' column found in the uploaded file.")

    def prepare_data(self):
        """Prepares the dataset for the Linear Regression model."""
        self.standardize_columns()
        self.process_date_column()
        self.df = self.df.sort_values('Date')
        self.df['Days'] = (self.df['Date'] - self.df['Date'].min()).dt.days
        return self.df


# Streamlit UI
st.title("📈 Stock Price Predictor")

# Sidebar File Uploader
st.sidebar.header("Upload Stock Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload your stock data file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processor = StockDataProcessor(df)
    df = processor.prepare_data()

    st.write("### Processed Data", df.head())

    # Selecting Features and Target
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
