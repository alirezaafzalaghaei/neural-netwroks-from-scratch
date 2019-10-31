import matplotlib.pyplot as plt
import os
import seaborn as sns

os.environ["KERAS_BACKEND"] = "theano"
from keras.datasets import fashion_mnist

from nn.mlp import MLP
from nn.mlp.activations import LeakyReLu

sns.set()


def load_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = load_mnist()

mlp = MLP([30, 10], activation=LeakyReLu(.1), batch_size=256, epochs=10, mu=0.95, beta=.2, eta=.5, alpha=.01,
          verbose=1, task='classification')

hist = mlp.fit(X_train, y_train)

acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

plt.plot(list(range(len(hist))), hist)
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
