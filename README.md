**LSTM-Based Stock Price Predictor**
This Streamlit app allows users to upload stock data (CSV) and uses Deep Learning (LSTM) to:
- Train an LSTM model on stock data (Close, Volume, Open, High, Low prices)
- Predict future stock prices & trading volume for 7, 15, 30, 90, or 180 days
- Show predictions both as a graph & table
- Let users fine-tune the model (LSTM units, dropout rate, epochs, batch size)

**Features Overview:**
- User Uploads CSV File (Stock Data)
- Preprocesses & Normalizes Data (Using MinMaxScaler)
- Trains an LSTM Model (Customizable via UI)
- Plots a Graph of Actual vs. Predicted Prices
- Displays a Table with Future Predictions (Date, Close Price, Volume)
- Fully Deployable on Render (With tensorflow-cpu for performance)

**HOW TO RUN:**

Access the link https://stock-price-predictor-4lgt.onrender.com

Upload .csv file with stock market data from this repository from https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
Or use the uploaded TSLA.csv dataset

*To note:
Render is a free instance which will spin down with inactivity, requests can be delayed by 50 seconds or more.
