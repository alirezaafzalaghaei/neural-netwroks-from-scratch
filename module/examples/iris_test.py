import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from nn.mlp import MLP
from nn.mlp.activations import Tanh

sns.set()

X, y = load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1))

mlp = MLP([5, 5, 5, 5], activation=Tanh(), batch_size=32, epochs=2000, mu=0.9, beta=1, eta=.1, alpha=.01,
          verbose=500, task='classification')

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
