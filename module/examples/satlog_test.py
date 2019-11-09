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

mlp = MLP([64], activation=ReLu(), batch_size=128, epochs=100, mu=0.95, beta=.3, eta=.02, alpha=.001,
          verbose=1, task='classification')

history, validation = mlp.fit(X_train, y_train, validation=[X_valid, y_valid])
valid0 = validation[:, 0]
hist0 = history[:, 0]

valid1 = validation[:, 1]
hist1 = history[:, 1]

acc_test = mlp.score(X_test, y_test)
acc_train = mlp.score(X_train, y_train)

print('train accuracy: %.2f%%' % (acc_train * 100))
print('test accuracy: %.2f%%' % (acc_test * 100))

f, (ax1, ax2) = plt.subplots(1, 2)
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
