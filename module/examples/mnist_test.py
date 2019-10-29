import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import zipfile

from nn.mlp import MLP
from nn.mlp.activations import LeakyReLu

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


X_train, X_test, y_train, y_test = load_mnist()

mlp = MLP([30, 10], activation=LeakyReLu(.1), batch_size=256, epochs=30, mu=0.95, beta=.2, eta=.5, alpha=.01,
          verbose=1, task='classification')

hist = mlp.fit(X_train, y_train)

acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('accuracy: %.2f%%' % (acc_train * 100))
print('accuracy: %.2f%%' % (acc_test * 100))

plt.plot(list(range(len(hist))), hist)
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
