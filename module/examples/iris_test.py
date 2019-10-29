import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from nn.mlp import MLP
from nn.mlp.activations import Tanh

sns.set()


def load_iris():
    iris = pd.read_csv('../datasets/iris.csv')
    X = iris.iloc[:, :-1].to_numpy()
    y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

    return X, y


X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLP([5, 5, 5, 5], activation=Tanh(), batch_size=32, epochs=2000, mu=0.9, beta=1, eta=.1, alpha=.01,
          verbose=500, task='classification')

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
