import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Leer el archivo CSV y almacenarlo en un DataFrame de pandas
data = pd.read_csv('/home/noe/GitHub/BTC/Dataset/final.csv')

# Definir el número de la columna objetivo (precio del bitcoin close adj)
target_col = 385

# Separar los datos en entrada (todas las columnas excepto la columna objetivo) y salida (columna objetivo)
x = data.iloc[:, :target_col]
y = data.iloc[:, target_col]

# Convertir los datos a un tensor de TensorFlow
x_tensor = tf.constant(x.values)
y_tensor = tf.constant(y.values.reshape(-1, 1))

# Normalizar los datos utilizando la función tf.keras.utils.normalize()
x_normalized = tf.keras.utils.normalize(x_tensor)

# Definir el tamaño del conjunto de datos y la proporción de entrenamiento/prueba
total_length = len(data)
train_size = int(total_length * 0.7)

# Crear los conjuntos de entrenamiento y prueba
x_train = x_normalized[:train_size, :]
y_train = y_tensor[:train_size, :]
x_test = x_normalized[train_size:, :]
y_test = y_tensor[train_size:, :]

# Imprimir la longitud de los conjuntos de entrenamiento y prueba
print("Número de ejemplos de entrenamiento:", len(x_train))
print("Número de ejemplos de prueba:", len(x_test))

# Definir el modelo para regresión
# Autoencoder
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=721, input_shape=[target_col], activation='relu'),
    tf.keras.layers.Dense(units=360, activation='relu'),
    tf.keras.layers.Dense(units=180, activation='relu'),
    tf.keras.layers.Dense(units=90, activation='relu'),
    tf.keras.layers.Dense(units=45, activation='relu'),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=45, activation='relu'),
    tf.keras.layers.Dense(units=90, activation='relu'),
    tf.keras.layers.Dense(units=180, activation='relu'),
    tf.keras.layers.Dense(units=360, activation='relu'),
    tf.keras.layers.Dense(units=target_col, activation='linear')
])

# Compile and fit autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=8, shuffle=False)

# Extract encoder part of autoencoder
encoder = tf.keras.models.Sequential(autoencoder.layers[:6])

# Obtain encoded features for training data
encoded_X_train = encoder.predict(x_train)

# tSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_features = tsne.fit_transform(encoded_X_train)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=721, input_shape=[target_col], kernel_regularizer=tf.keras.regularizers.l2(0.05)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=360),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=180),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=90),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=45),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=20),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='linear')
])


model.summary()

batch_size = 16

# Definir callbacks
checkpoint_callback = ModelCheckpoint('/home/noe/GitHub/BTC/Models/model-btc.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='min')

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        mode='min',
                                        restore_best_weights=True)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=500, validation_data=(x_test, y_test),
                    callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])

# Evaluar el modelo en el conjunto de prueba
loss, mse = model.evaluate(x_test, y_test)
print('Mean Squared Error: {}'.format(mse))