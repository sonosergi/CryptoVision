import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("/home/noe/GitHub/BTC/FinanceData/Dataset/original-dataset.csv").drop(index=range(3200))
print(dataset.isnull().sum().sum())
# Iterar por cada fila del DataFrame
for col in dataset.columns:
    null_values = dataset[col].isnull().sum()
    if null_values > 0:
        print(f"Columna {col}: {null_values} valores nulos")
        for i in range(len(dataset)):
            if pd.isna(dataset[col][i]):
                prev_mean = dataset[col].iloc[max(0, i-20):i].rolling(window=20, min_periods=1).mean().iloc[-1]
                next_mean = dataset[col].iloc[i+1:min(i+21, len(dataset))].rolling(window=20, min_periods=1).mean().iloc[-1]
                if i < 20:
                    prev_mean = dataset[col].iloc[0:i].mean()
                else:
                    prev_mean = dataset[col].iloc[i-20:i].rolling(window=20, min_periods=1).mean().iloc[-1]

                dataset[col][i] = (prev_mean + next_mean) / 2
        print(f"Columna {col}: {dataset[col].isnull().sum()} valores nulos restantes")


"""
# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test = train_test_split(dataset, test_size=0.3, random_state=False)


# Definir el número de neuronas en la capa de entrada, la capa oculta y la capa de salida
input_dim = 200
hidden_dim = 100

# Definir la entrada
inputs = Input(shape=(input_dim,))

# Definir el codificador
encoder = Dense(hidden_dim, activation='relu')(inputs)

# Definir el decodificador
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Crear el modelo autoencoder
autoencoder = Model(inputs, decoder)

# Compilar el modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Crear un objeto TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Entrenar el modelo con el objeto TensorBoard
autoencoder.fit(X_train, X_test, epochs=10, callbacks=[tensorboard], batch_size=32, validation_data=(X_test, X_test))

autoencoder.save('autoencoder-NaN.h5')
"""