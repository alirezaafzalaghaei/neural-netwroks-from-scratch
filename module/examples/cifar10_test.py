import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow').setLevel(logging.Error)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras
import matplotlib.pyplot as plt
from nn.cnn.cnn import CNN
import numpy as np
from keras.layers import *
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = np.unique(y_train).shape[0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_size = 16
epochs = 3
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
arch = [Conv2D(32, (3, 3), padding='same', activation='relu'),Flatten(),Dense(32, activation='relu')]

model = CNN(arch, epochs, batch_size, optimizer, loss, metrics, task='classification')
history = model.fit(x_train, y_train, validation_split=.2)
print(model.score(x_test, y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
