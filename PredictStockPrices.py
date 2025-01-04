import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Fetch stock data using yfinance
ticker = "AAPL"  # Replace with the stock ticker you want (e.g., 'GOOGL', 'MSFT')
data = yf.download(ticker, start="2010-01-01", end="2023-12-31")  # Adjust date range as needed
prices = data['Close'].values.reshape(-1, 1)  # Use 'Close' prices

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create a function to prepare data
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Hyperparameters
sequence_length = 60  # Use the last 60 days to predict the next day

# Prepare data
x, y = create_sequences(prices_scaled, sequence_length)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # Reshape for LSTM input
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))  # De-normalize predictions
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))  # De-normalize actual values

# Plot predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()
