import os
import texttable

os.environ["KERAS_BACKEND"] = "theano"
import keras
from nn.cnn.grid_search import CNNGridSearch
import numpy as np
from keras.layers import *
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# img_rows, img_cols = 28, 28
# import keras.backend as K
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255

num_classes = np.unique(y_train).shape[0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

archs = [[Conv2D(8, (3, 3), padding='same', activation='relu'), Flatten()],
         [Conv2D(16, (3, 3), padding='same', activation='relu'), Flatten(), Dense(32, activation='relu')]
         ]
batch_sizes = [2048]
optimizers = [keras.optimizers.Adam()]
epochs = [2]
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
