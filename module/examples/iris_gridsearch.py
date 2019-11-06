import time

import seaborn as sns
import texttable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from nn.mlp import MLPGridSearch
from nn.mlp.activations import *

sns.set()

hidden_layers = [(5, 5, 5, 5), (10, 10, 5), (15, 15), (20, 15, 10)]
activations = [Tanh(), LeakyReLu(.1), ReLu()]
batch_sizes = [16, 32, 64]
epochs = [50]
mus = [0.85, 0.9, 0.95]
betas = [.1, .2]
etas = [.01, .1, 0.001]
alphas = [.001, 0.01, .1]

X, y = load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1))
t = time.time()
mlp = MLPGridSearch('classification', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas,
                    'iris.csv')
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
