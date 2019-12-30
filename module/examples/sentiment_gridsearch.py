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
from nn.rnn.gridsearch import RNNGridSearch
import numpy as np
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# load dataset into a list of strings

with open('../datasets/sentiment-data/reviews.txt', 'r') as f:
    reviews = f.read().strip().split('\n')
with open('../datasets/sentiment-data/labels.txt', 'r') as f:
    labels = f.read().strip().split('\n')

skip_common = 25
max_dict_words = 300
oov_char = 2

tokenizer = Tokenizer(num_words=max_dict_words + skip_common)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(
    reviews)  # convert comment words to integers, sorted by count

sequences = [[w if w > skip_common else oov_char for w in seq]
             for seq in sequences]  # skip common words

max_sent_words = 300

X = pad_sequences(sequences, max_sent_words)
Y = np.array([label == 'positive' for label in labels])
Y = keras.utils.to_categorical(Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# Conv1D(4, 3, padding='valid', activation='relu', strides=1)
# GlobalMaxPooling1D()
# Dense(16,activation='relu')

archs = [
    [Embedding(max_dict_words + skip_common, 10, input_length=max_sent_words), LSTM(8)],
    [Embedding(max_dict_words + skip_common, 50, input_length=max_sent_words), LSTM(8)],
    [Embedding(max_dict_words + skip_common, 100, input_length=max_sent_words), LSTM(8)],
    [Embedding(max_dict_words + skip_common, 200, input_length=max_sent_words), LSTM(8)],
    [Embedding(max_dict_words + skip_common, 500, input_length=max_sent_words), LSTM(8)],
    [Embedding(max_dict_words + skip_common, 1000, input_length=max_sent_words), LSTM(8)],

    [Embedding(max_dict_words + skip_common, 10, input_length=max_sent_words), LSTM(16)],
    [Embedding(max_dict_words + skip_common, 50, input_length=max_sent_words), LSTM(16)],
    [Embedding(max_dict_words + skip_common, 100, input_length=max_sent_words), LSTM(16)],
    [Embedding(max_dict_words + skip_common, 200, input_length=max_sent_words), LSTM(16)],
    [Embedding(max_dict_words + skip_common, 500, input_length=max_sent_words), LSTM(16)],
    [Embedding(max_dict_words + skip_common, 1000, input_length=max_sent_words), LSTM(16)],

    [Embedding(max_dict_words + skip_common, 10, input_length=max_sent_words), LSTM(64)],
    [Embedding(max_dict_words + skip_common, 50, input_length=max_sent_words), LSTM(64)],
    [Embedding(max_dict_words + skip_common, 100, input_length=max_sent_words), LSTM(64)],
    [Embedding(max_dict_words + skip_common, 200, input_length=max_sent_words), LSTM(64)],
    [Embedding(max_dict_words + skip_common, 500, input_length=max_sent_words), LSTM(64)],
    [Embedding(max_dict_words + skip_common, 1000, input_length=max_sent_words), LSTM(64)],
]
batch_sizes = [256]
optimizers = [keras.optimizers.Adam()]
epochs = [15]
model = RNNGridSearch(archs, epochs, batch_sizes, optimizers, 'sentiment1.csv')
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
