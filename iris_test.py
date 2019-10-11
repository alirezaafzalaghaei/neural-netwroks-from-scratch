import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import MLP

np.random.seed(1)

iris = pd.read_csv('./datasets/iris.csv')
X = iris.iloc[:, :-1].to_numpy()
y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)
X = StandardScaler().fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

mlp = MLP([5, 5, 5, 5], epochs=1000, beta=1, eta=.5)
hist = mlp.run(Xtrain, ytrain)
ytr_p = mlp.predict(Xtrain)
yte_p = mlp.predict(Xtest)

print('train loss: %.4f' % hist[-1])
print('test  loss: %.4f' % mlp.loss(ytest, yte_p))
c_te = ytest - yte_p
c_tr = ytrain - ytr_p
c_te = c_te[np.abs(c_te) > .05]
c_tr = c_tr[np.abs(c_tr) > .05]
acc_tr = 100 * (1 - c_tr.shape[0] / ytrain.shape[0])
acc_te = 100 * (1 - c_te.shape[0] / ytest.shape[0])
print('train accuracy: %.2f%%' % acc_tr)
print('test  accuracy: %.2f%%' % acc_te)

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("%.2e" % hist[-1])
plt.show()
