# LSTM Stock Price Predictor

This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock closing prices based on historical data. It leverages the yfinance library to fetch real-time stock data, preprocesses it, and trains a deep learning model to make predictions.

## Features

  - Fetch historical stock data using yfinance.

  - Preprocess and normalize data for training.

  - Build and train an LSTM model using TensorFlow/Keras.

  - Visualize training performance and prediction results.

## Requirements

To run this project, install the following Python packages:

`pip install tensorflow numpy pandas scikit-learn matplotlib yfinance`

## How to Use

1. Clone or Download the Project:

     - Copy the script to your local machine.

2. Install Dependencies:

    - Use the pip install commands listed above to install the required packages.

3. Run the Code:

    - Open the script in your preferred Python IDE (e.g., Thonny, VSCode).

    - Execute the script to start training the model and predicting stock prices.

4. Customize Parameters:

    - Change the stock ticker symbol (e.g., AAPL for Apple, GOOGL for Alphabet) in the ticker variable.

    - Adjust start and end dates to fetch different historical data.

    - Modify hyperparameters such as sequence_length, epochs, and batch_size as needed.

## Outputs

1. Training and Validation Loss Plot:

    - Visualizes how the model's loss decreases during training.

2. Predicted vs. Actual Prices Plot:

    - Compares the predicted stock prices with actual prices from the test dataset.

## Example Workflow

1. Fetch historical data for a specific stock:

`ticker = "AAPL"  # Replace with your desired stock ticker
data = yf.download(ticker, start="2010-01-01", end="2023-12-31")`

2. Preprocess data and create training sequences.

3. Train an LSTM model:

   - The model uses the last 60 days of data to predict the next day's closing price.

4. Visualize predictions:

   - Generate graphs to compare the predicted prices with the actual prices.

## Customization

- Sequence Length:

  - Adjust the sequence_length variable to change how many days of historical data the model uses for prediction.

- Model Architecture:

  - Modify the number of LSTM units or add layers to improve performance.

- Optimizer and Loss Function:

  - Experiment with different optimizers (e.g., SGD, RMSprop) and loss functions to achieve better results.

## Notes

- Ensure a stable internet connection to fetch data from yfinance.

- Larger datasets or longer sequences may require more training time and computational resources.

## License

This project is for educational purposes. You are free to modify and use it as needed.

## Contact

For any issues or questions, feel free to reach out.
