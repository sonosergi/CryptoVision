# CryptoVision
This is a deep learning model trained using Tensorflow that predicts the adjusted closing price of Bitcoin based on the historical data of 103 assets, their moving averages, stochastic values, and option prices calculated using the Black-Scholes method.

## Dataset
The dataset is composed of 103 assets with their respective indicators. The model's goal is to predict Bitcoin's adjusted closing price based on the historical data of all these assets.
This code is written in Python and uses the following libraries: Pandas, NumPy, scikit-learn (specifically the t-SNE implementation), and TensorFlow.

## Steps
The steps involved are:

* Import the necessary libraries:

    pandas for reading and manipulating data.
    numpy for numerical computing.
    sklearn.manifold.TSNE for t-SNE dimensionality reduction.
    tensorflow for building and training machine learning models.
    tensorflow.keras.callbacks for specifying callbacks during model training.

* Read in the CSV file using pandas and store it as a DataFrame.

* Define the target column number (the column containing the Bitcoin price) and separate the input (all columns except the target column) and output (target column) data.

* Convert the data to TensorFlow tensors.

* Normalize the input data using the tf.keras.utils.normalize() function.

* Define the size of the dataset and the training/testing split proportion.

* Create the training and testing datasets using the train_test_split() function.

* Define the autoencoder model using the tf.keras.models.Sequential() function and add layers to it.

* The autoencoder model has a series of dense layers with different numbers of neurons and activation functions. The input layer has 721 neurons, the output layer has the same number of neurons as the input layer, and the remaining layers have 360, 180, 90, 45, 20, 45, 90, 180, and 360 neurons, respectively.

* The activation function for all layers except the output layer is ReLU (rectified linear unit), while the output layer uses a linear activation function.

* The autoencoder model is now ready for training. No training is performed in this code block.

## Training

The model was trained using 80% of the dataset for training and 20% for validation, with a batch size of 64 and a learning rate of 0.001. The model was trained for 100 epochs, and the best model was selected based on the validation loss.
