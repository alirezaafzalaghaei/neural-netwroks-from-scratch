import matplotlib.pyplot as plt
import os
import seaborn as sns

os.environ["KERAS_BACKEND"] = "theano"
from keras.datasets import fashion_mnist

from nn.mlp import MLP
from nn.mlp.activations import *
from sklearn.model_selection import train_test_split

sns.set()


def load_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


X, X_test, y, y_test = load_mnist()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)


mlp = MLP([128], activation=ReLu(), batch_size=64, epochs=50, mu=0.95, beta=.2, eta=.4, alpha=.001,
          verbose=1, task='classification')

hist, valid = mlp.fit(X_train, y_train, validation=(X_valid, y_valid))
valid = valid[:, 1]
hist = hist[:, 1]
acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

plt.plot(list(range(len(hist))), hist, label='train')
plt.plot(list(range(len(hist))), valid, label='valid')
plt.legend()
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
