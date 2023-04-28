# CryptoVision
This is a deep learning model trained using Tensorflow that predicts the adjusted closing price of Bitcoin based on the historical data of 103 assets, their moving averages, stochastic values, and option prices calculated using the Black-Scholes method.
## Dataset

The dataset is composed of 103 assets with their respective indicators. The model's goal is to predict Bitcoin's adjusted closing price based on the historical data of all these assets.

## Training

The model was trained using 80% of the dataset for training and 20% for validation, with a batch size of 64 and a learning rate of 0.001. The model was trained for 100 epochs, and the best model was selected based on the validation loss.
