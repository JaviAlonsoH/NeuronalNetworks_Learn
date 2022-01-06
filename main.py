
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")

historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Predicciones: ")
resultado = modelo.predict([100.0])
print("El resultado de la predicción es: " + str(resultado) + " fahrenheit! ")

print("Variables internas del modelo")
print(capa.get_weights())