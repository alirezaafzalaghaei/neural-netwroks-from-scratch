import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import texttable
from sklearn.model_selection import train_test_split

from nn.mlp import MLPGridSearch
from nn.mlp.activations import LeakyReLu, Tanh

sns.set()


def load_mnist():
    # with zipfile.ZipFile("../datasets/mnist/mnist_train.zip", "r") as zip_file:
    #     zip_file.extractall('../datasets/mnist/')
    # with zipfile.ZipFile("../datasets/mnist/mnist_test.zip", "r") as zip_file:
    #     zip_file.extractall('../datasets/mnist/')
    # train_data = np.loadtxt("../datasets/mnist/mnist_train.csv", delimiter=",")
    # test_data = np.loadtxt("../datasets/mnist/mnist_test.csv", delimiter=",")
    # np.save('../datasets/mnist/train',train_data)
    # np.save('../datasets/mnist/test', test_data)
    # os.remove('../datasets/mnist/mnist_train.csv')
    # os.remove('../datasets/mnist/mnist_test.csv')
    train_data = np.load('../datasets/mnist/train.npy')
    test_data = np.load('../datasets/mnist/test.npy')
    X_train, y_train = train_data[:, 1:], train_data[:, 0].reshape(-1, 1)
    X_test, y_test = test_data[:, 1:], test_data[:, 0].reshape(-1, 1)
    return X_train, X_test, y_train, y_test




hidden_layers = [(5, 5, 5, 5)]
activations = [Tanh()]
batch_sizes = [32]
epochs = [3]
mus = [0.95]
betas = [.1]
etas = [.01,.4]
alphas = [.001]

X_train, X_test, y_train, y_test = load_mnist()
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
