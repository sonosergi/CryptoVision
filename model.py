import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


data = pd.read_csv('/home/noe/GitHub/BTC/Dataset/final-dataset.csv')

"""
# buscar el índice de la posición donde se encuentra el valor 16602.5859375 correspondiente al precio de cierre adj del btc
row_index, col_index = np.where(data == 16602.5859375)

# imprimir el índice de fila y columna donde se encuentra el valor
print("El valor 16602.5859375 se encuentra en la posición: ({}, {})".format(row_index[0], col_index[0]))
"""

# Corresponde al Close Adj del BTC
y = data.iloc[:, 719].values
X = data.iloc[:, data.columns != 'columna719'].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=164, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(units=320, activation='relu'),
    tf.keras.layers.Dense(units=640, activation='relu'),
    tf.keras.layers.Dense(units=320, activation='relu'),
    tf.keras.layers.Dense(units=164, activation='relu'),
    tf.keras.layers.Dense(units=1)
])


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='mean_squared_error', # corrected loss function
              metrics=['accuracy'] # corrected metric
             )

model.summary()

# Definir callbacks
checkpoint_callback = ModelCheckpoint('/home/noe/GitHub/BTC/Models/best_model.h5',
                                       monitor='val_mean_squared_error', # corrected monitor
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max') # corrected mode

early_stopping_callback = EarlyStopping(monitor='accuracy', # corrected monitor
                                        patience=25,
                                        mode='max', # corrected mode
                                        restore_best_weights=True)

# Entrenar modelo
history = model.fit(X_train,
                    y_train, # corrected input parameter
                    validation_data=(X_test, y_test), # corrected input parameter
                    epochs=5000,
                    callbacks=[early_stopping_callback, checkpoint_callback])

test_loss = model.evaluate(X_test, y_test)





