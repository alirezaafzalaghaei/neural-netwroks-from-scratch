import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow').setLevel(logging.Error)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import texttable
import keras
from nn.cnn.grid_search import CNNGridSearch
import numpy as np
from keras.layers import *
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = np.unique(y_train).shape[0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

archs = [[Conv2D(8, (3, 3), padding='same', activation='relu'), Flatten(), Dense(32, activation='relu')],
         [Conv2D(16, (3, 3), padding='same', activation='relu'), Flatten(), Dense(32, activation='relu')]
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
