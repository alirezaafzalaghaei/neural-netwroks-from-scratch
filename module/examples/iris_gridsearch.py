import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import texttable
from sklearn.model_selection import train_test_split

from nn.mlp import MLPGridSearch
from nn.mlp.activations import LeakyReLu, Tanh

sns.set()


def load_iris():
    iris = pd.read_csv('../datasets/iris.csv')
    X = iris.iloc[:, :-1].to_numpy()
    y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

    return X, y


hidden_layers = [(5, 5, 5, 5), (10, 10, 5), (15, 15)]
activations = [Tanh(), LeakyReLu(.1)]
batch_sizes = [32, 64]
epochs = [300]
mus = [0.95, .9]
betas = [.1, .2]
etas = [.01]
alphas = [.001, .1]

X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y)
t = time.time()
mlp = MLPGridSearch('classification', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas)
histories = mlp.run(X_train, y_train, X_test, y_test)
t = time.time() - t
print('time taken = %.2f sec' % t)

result = mlp.best_model()
hist = result.pop('history')
print('Best model is: ')

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
