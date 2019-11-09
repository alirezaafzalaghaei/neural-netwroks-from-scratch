import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from nn.mlp import MLP
from nn.mlp.activations import *

sns.set()

X, y = load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1))

mlp = MLP([10], activation=LeakyReLu(), batch_size=16, epochs=600, mu=.9, beta=.1, eta=.01, alpha=.001, verbose=500,
          task='classification')

hist, _ = mlp.fit(X_train, y_train, validation=[X_test, y_test])
hist = hist[:, 0]
acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

plt.plot(list(range(len(hist))), hist)
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
