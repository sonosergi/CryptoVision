import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("/home/noe/GitHub/BTC/FinanceData/Dataset/original-dataset.csv").drop(index=np.arange(3600))
print(dataset.isnull().sum().sum())
print(dataset.shape)
print(dataset.info)

print(dataset.isnull().sum().sum())



# Reemplazar los valores NaN por el promedio de la columna correspondiente
for col in dataset.columns:
    col_mean = dataset[col].mean()
    dataset[col] = dataset[col].fillna(col_mean)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test = train_test_split(dataset, test_size=0.3, random_state=False)

from sklearn.preprocessing import StandardScaler

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el número de neuronas en la capa de entrada, la capa oculta y la capa de salida
input_dim = X_train_scaled.shape[1]
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
autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                     )

# Imputar los valores faltantes
X_train_imputed = autoencoder.predict(X_train_scaled)
X_test_imputed = autoencoder.predict(X_test_scaled)

# Crear un objeto TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Entrenar el modelo utilizando el callback
autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=2500,
                batch_size=15,
                validation_data=(X_test_scaled, X_test_scaled),
                callbacks=[tensorboard])

autoencoder.save('autoencoder/autoencoder-NaN.h5')
