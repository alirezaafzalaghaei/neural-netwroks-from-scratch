import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from nn.mlp import MLP
from nn.mlp.activations import *

sns.set()


def load_satlog():
    with zipfile.ZipFile('../datasets/satlog/satlog.zip', 'r') as zip_ref:
        zip_ref.extractall('../datasets/satlog/')
    train = np.loadtxt('../datasets/satlog/sat.trn', delimiter=' ')
    test = np.loadtxt('../datasets/satlog/sat.tst', delimiter=' ')
    os.remove('../datasets/satlog/sat.trn')
    os.remove('../datasets/satlog/sat.tst')
    X_train, y_train = train[:, :-1], train[:, -1].reshape(-1, 1)
    X_test, y_test = test[:, :-1], test[:, -1].reshape(-1, 1)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_satlog()

mlp = MLP([15, 15, 15], activation=LeakyReLu(.03), batch_size=256, epochs=1500, mu=0.95, beta=.6, eta=.2, alpha=.01,
          verbose=50, task='classification')

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
