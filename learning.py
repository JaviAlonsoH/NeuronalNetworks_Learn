import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
print(train_labels[6])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # activation function ("relu") is "rectified linear unit"
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train model
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict([test_images[7]])

for x in range(5):
    plt.grid(False)
    plt.imshow(test_images[x], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + test_labels[x])
    plt.title("Prediction " + class_names[np.argmax(prediction[x])])
    plt.show()