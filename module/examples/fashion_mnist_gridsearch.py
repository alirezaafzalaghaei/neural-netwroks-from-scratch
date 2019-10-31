import time

import matplotlib.pyplot as plt
import seaborn as sns
import texttable
import os

os.environ["KERAS_BACKEND"] = "theano"
from keras.datasets import fashion_mnist

from nn.mlp import MLPGridSearch
from nn.mlp.activations import *

sns.set()


def load_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


hidden_layers = [(10, 10), (20, 10), (30, 20, 10), (10, 10, 10, 10), (100,)]
activations = [Tanh(), LeakyReLu(0.02), ReLu()]
batch_sizes = [512, 2048]
epochs = [300]
mus = [0.95, .9]
betas = [.1, .2]
etas = [.001, .1, .25]
alphas = [.001, 0.01, .1]

X_train, X_test, y_train, y_test = load_mnist()
t = time.time()
mlp = MLPGridSearch('classification', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas,
                    'fashion.csv')
histories = mlp.run(X_train, y_train, X_test, y_test)
t = time.time() - t
print('time taken = %s seconds' % time.strftime('%H:%M:%S', time.gmtime(t)))

result = mlp.best_model()
hist = result.pop('history')
print('Best model is: ')

tbl = texttable.Texttable()
tbl.set_cols_align(["c", "c"])
tbl.set_cols_valign(["c", "c"])
tbl.add_rows([['Hyperparameter', 'Best value'], *list(result.items())])
print(tbl.draw())
