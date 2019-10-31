import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import texttable
import os
import zipfile
from nn.mlp import MLPGridSearch
from nn.mlp.activations import *

sns.set()


def load_satlog():
    with zipfile.ZipFile('../datasets/satlog/satlog.zip', 'r') as zip_ref:
        zip_ref.extractall('../datasets/satlog/')
    train = np.loadtxt('../datasets/satlog/sat.trn', delimiter=' ')
    test = np.loadtxt('../datasets/satlog/sat.tst', delimiter=' ')
    os.remove('../datasets/satlog/sat.trn')
    os.remove('../datasets/satlog/sat.tst')
    X_train, y_train = train[:, :-1], train[:, -1].reshape(-1, 1)
    X_test, y_test = test[:, :-1], test[:, -1].reshape(-1, 1)
    return X_train, X_test, y_train, y_test


hidden_layers = [(32, 16, 8), (10, 10), (12, 6)]
activations = [Tanh(), Sigmoid(), LeakyReLu(.03)]
batch_sizes = [64, 256, 512]
epochs = [1000]
mus = [0.95, .9]
betas = [.1, .3]
etas = [.01, 0.1, .3]
alphas = [.001, 0.01, 0.1]

X_train, X_test, y_train, y_test = load_satlog()
t = time.time()
mlp = MLPGridSearch('classification', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas,
                    'satlog.csv')
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
