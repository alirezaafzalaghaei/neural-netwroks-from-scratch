import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from nn.mlp import MLP
from nn.mlp.activations import *
from sklearn.model_selection import train_test_split

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
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

mlp = MLP([32, 16, 8], activation=ReLu(), batch_size=128, epochs=250, mu=0.9, beta=.2, eta=.02, alpha=.001,
          verbose=5, task='classification')

hist, valid = mlp.fit(X_train, y_train, validation=[X_valid, y_valid])
valid = valid[:, 0]
hist = hist[:, 0]

acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

plt.plot(list(range(len(hist))), hist, label='train')
plt.plot(list(range(len(hist))), valid, label='valid')
plt.legend()
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()
