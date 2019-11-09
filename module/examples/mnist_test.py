from nn.mlp.activations import *
from nn.mlp import MLP
import os

os.environ["KERAS_BACKEND"] = "theano"
from keras.datasets import mnist
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = load_mnist()

mlp = MLP([128, 64], activation=ReLu(), batch_size=64, epochs=15, mu=0.85, beta=.1, eta=.03, alpha=.001,
          verbose=1, task='classification')

history, validation = mlp.fit(X_train, y_train, validation=[X_test, y_test])
valid0 = validation[:, 0]
hist0 = history[:, 0]

valid1 = validation[:, 1]
hist1 = history[:, 1]

acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

f, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(list(range(len(hist0))), hist0, label='train')
ax1.plot(list(range(len(hist0))), valid0, label='valid')
ax1.legend()
ax1.set_xlabel('iterations')
ax1.set_ylabel('Loss')
ax1.set_title('loss')

ax2.plot(list(range(len(hist0))), hist1, label='train')
ax2.plot(list(range(len(hist0))), valid1, label='valid')
ax2.legend()
ax2.set_xlabel('iterations')
ax2.set_ylabel('accuracy')
ax2.set_title('accuracy')
plt.show()
