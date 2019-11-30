import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras.backend as K
import texttable
import keras
from nn.cnn.grid_search import CNNGridSearch
import numpy as np
from keras.layers import *
from keras.datasets import fashion_mnist as cifar10

img_rows = img_cols = 28
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
num_classes = np.unique(y_train).shape[0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

archs = [[Conv2D(8, (3, 3), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (3, 3), padding='same', activation='relu'), Flatten()],
         [Conv2D(32, (3, 3), padding='same', activation='relu'), Flatten()],
         [Conv2D(64, (3, 3), padding='same', activation='relu'), Flatten()],
         [Conv2D(128, (3, 3), padding='same', activation='relu'), Flatten()],

         [Conv2D(8, (5, 5), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (5, 5), padding='same', activation='relu'), Flatten()],
         [Conv2D(32, (5, 5), padding='same', activation='relu'), Flatten()],
         [Conv2D(64, (5, 5), padding='same', activation='relu'), Flatten()],
         [Conv2D(128, (5, 5), padding='same', activation='relu'), Flatten()],

         [Conv2D(8, (7, 7), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (7, 7), padding='same', activation='relu'), Flatten()],
         [Conv2D(32, (7, 7), padding='same', activation='relu'), Flatten()],
         [Conv2D(64, (7, 7), padding='same', activation='relu'), Flatten()],
         [Conv2D(128, (7, 7), padding='same', activation='relu'), Flatten()],

         [Conv2D(8, (9, 9), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (9, 9), padding='same', activation='relu'), Flatten()],
         [Conv2D(32, (9, 9), padding='same', activation='relu'), Flatten()],
         [Conv2D(64, (9, 9), padding='same', activation='relu'), Flatten()],
         [Conv2D(128, (9, 9), padding='same', activation='relu'), Flatten()],

         [Conv2D(8, (11, 11), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (11, 11), padding='same', activation='relu'), Flatten()],
         [Conv2D(32, (11, 11), padding='same', activation='relu'), Flatten()],
         [Conv2D(64, (11, 11), padding='same', activation='relu'), Flatten()],
         [Conv2D(128, (11, 11), padding='same', activation='relu'), Flatten()],

         [Conv2D(8, (3, 3), padding='valid', activation='relu'), Flatten()],
         [Conv2D(16, (3, 3), padding='valid', activation='relu'), Flatten()],
         [Conv2D(32, (3, 3), padding='valid', activation='relu'), Flatten()],
         [Conv2D(64, (3, 3), padding='valid', activation='relu'), Flatten()],
         [Conv2D(128, (3, 3), padding='valid', activation='relu'), Flatten()],

         [Conv2D(8, (5, 5), padding='valid', activation='relu'), Flatten()],
         [Conv2D(16, (5, 5), padding='valid', activation='relu'), Flatten()],
         [Conv2D(32, (5, 5), padding='valid', activation='relu'), Flatten()],
         [Conv2D(64, (5, 5), padding='valid', activation='relu'), Flatten()],
         [Conv2D(128, (5, 5), padding='valid', activation='relu'), Flatten()],

         [Conv2D(8, (7, 7), padding='valid', activation='relu'), Flatten()],
         [Conv2D(16, (7, 7), padding='valid', activation='relu'), Flatten()],
         [Conv2D(32, (7, 7), padding='valid', activation='relu'), Flatten()],
         [Conv2D(64, (7, 7), padding='valid', activation='relu'), Flatten()],
         [Conv2D(128, (7, 7), padding='valid', activation='relu'), Flatten()],

         [Conv2D(8, (9, 9), padding='valid', activation='relu'), Flatten()],
         [Conv2D(16, (9, 9), padding='valid', activation='relu'), Flatten()],
         [Conv2D(32, (9, 9), padding='valid', activation='relu'), Flatten()],
         [Conv2D(64, (9, 9), padding='valid', activation='relu'), Flatten()],
         [Conv2D(128, (9, 9), padding='valid', activation='relu'), Flatten()],

         [Conv2D(8, (11, 11), padding='valid', activation='relu'), Flatten()],
         [Conv2D(16, (11, 11), padding='valid', activation='relu'), Flatten()],
         [Conv2D(32, (11, 11), padding='valid', activation='relu'), Flatten()],
         [Conv2D(64, (11, 11), padding='valid', activation='relu'), Flatten()],
         [Conv2D(128, (11, 11), padding='valid', activation='relu'), Flatten()],
         ]
batch_sizes = [128]
optimizers = [keras.optimizers.Adam()]
epochs = [20]
model = CNNGridSearch(archs, epochs, batch_sizes, optimizers, 'cifar10_cnn.csv')
histories = model.run(x_train, y_train, x_test, y_test)

result = model.best_model()
hist = result.pop('history_loss')
result.pop('history_score')
print('Best model is: ')

tbl = texttable.Texttable()
tbl.set_cols_align(["c", "c"])
tbl.set_cols_valign(["c", "c"])
tbl.add_rows([['Hyperparameter', 'Best value'], *list(result.items())])
print(tbl.draw())
