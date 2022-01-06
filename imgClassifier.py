import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import math
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

data_training, data_test = datos['train'], datos['test']

name_classes = metadatos.features['label'].names

print(name_classes)

#Normalize data (pass colors from 0-255 to 0-1)

def normalize(imgs, labels):
    imgs = tf.cast(imgs, tf.float32)
    imgs /= 255
    return imgs, labels

#Normalize training data

data_training = data_training.map(normalize)
data_test = data_test.map(normalize)

#Add to cache

data_training = data_training.cache()
data_test = data_test.cache()

for img, label in data_training.take(1):
    break
img = img.numpy().reshape((28,28)) #redimension

#Draw

plt.figure()
plt.imshow(img, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i, (img, label) in enumerate(data_training.take(25)):
    img = img.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(name_classes[label])

plt.show()

#Create model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), #1 - black & white
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # for classification network
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_ex_training = metadatos.splits['train'].num_examples
num_ex_test = metadatos.splits['test'].num_examples

print(num_ex_training)
print(num_ex_test)


STACK_SIZE = 32

data_training = data_training.repeat().shuffle(num_ex_training).batch(STACK_SIZE)
data_test = data_training.batch(STACK_SIZE)

#Training

history = model.fit(data_training, epochs=5, steps_per_epoch=math.ceil(num_ex_training/STACK_SIZE))

plt.xlabel("#Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])


## Tests ##

for imgs_test, labels_test, in data_test.take(1):
    imgs_test = imgs_test.numpy()
    labels_test = labels_test.numpy()
    predic = model.predict(imgs_test)

def graphic_image(i, arr_predicts, real_labels, imgs):
    arr_predicts, real_labels, img = arr_predicts[i], real_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    prediction_label = np.argmax(arr_predicts)
    if prediction_label == real_labels:
        color = 'blue' #if hits
    else:
        color = 'red' #if it misses

    plt.xlabel("{} {:2.0f}% ({})".format(
        name_classes[prediction_label],
        100*np.max(arr_predicts),
        name_classes[real_labels],
        color=color
    ))

def graphic_value(i, arr_predicts, real_label):
    arr_predicts, real_label = arr_predicts[i], real_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    graphic = plt.bar(range(10), arr_predicts, color="#777777")
    plt.ylim([0,1])
    prediction_label = np.argmax(arr_predicts)
    graphic[prediction_label].set_color('red')
    graphic[real_label].set_color('blue')

rows = 5
columns = 5
num_imgs = rows*columns
plt.figure(figsize=(2*2*columns, 2*rows))
for i in range(num_imgs):
    plt.subplot(rows, 2*columns, 2*i+1)
    graphic_image(i, predic, labels_test, imgs_test)
    plt.subplot(rows, 2*columns, 2*i+2)
    graphic_value(i, predic, labels_test)