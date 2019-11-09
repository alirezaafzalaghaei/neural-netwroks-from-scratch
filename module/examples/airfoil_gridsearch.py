import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import texttable
from sklearn.model_selection import train_test_split

from nn.mlp import MLPGridSearch
from nn.mlp.activations import LeakyReLu, Tanh

sns.set()


def load_airfoil():
    data_set = np.loadtxt('../datasets/airfoil.dat')
    X = data_set[:, :-1]
    y = data_set[:, -1].reshape(-1, 1)

    return X, y


hidden_layers = [(5, 5), (10, 10)]
activations = [Tanh(), LeakyReLu(.05)]
batch_sizes = [128]
epochs = [100]
mus = [0.85]
betas = [.06]
etas = [.01, 0.06]
alphas = [.001, .01]

X, y = load_airfoil()
X_train, X_test, y_train, y_test = train_test_split(X, y)
t = time.time()
mlp = MLPGridSearch('regression', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas, 'airfoil.csv')
histories = mlp.run(X_train, y_train, X_test, y_test)
t = time.time() - t
print('time taken = %.2f sec' % t)

result = mlp.best_model()
hist = result.pop('history_loss')
result.pop('history_score')

tbl = texttable.Texttable()
tbl.set_cols_align(["c", "c"])
tbl.set_cols_valign(["c", "c"])
tbl.add_rows([['Hyperparameter', 'Best value'], *list(result.items())])
print(tbl.draw())

plt.plot(list(range(len(hist))), hist)
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
