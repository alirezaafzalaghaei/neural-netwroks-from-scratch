import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import texttable


from nn.mlp import MLPGridSearch
from nn.mlp.activations import LeakyReLu, Tanh

sns.set()


def load_satlog():
    train = np.loadtxt('../datasets/satlog/sat.trn', delimiter=' ')
    test = np.loadtxt('../datasets/satlog/sat.tst', delimiter=' ')
    X_train, y_train = train[:, :-1], train[:, -1].reshape(-1, 1)
    X_test, y_test = test[:, :-1], test[:, -1].reshape(-1, 1)
    return X_train, X_test, y_train, y_test


hidden_layers = [(5, 5, 5, 5), (10, 10, 5), (15, 15)]
activations = [Tanh(), LeakyReLu(.1)]
batch_sizes = [32]
epochs = [30]
mus = [0.95, .9]
betas = [.1]
etas = [.01, .3]
alphas = [.001]

X_train, X_test, y_train, y_test = load_satlog()
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
