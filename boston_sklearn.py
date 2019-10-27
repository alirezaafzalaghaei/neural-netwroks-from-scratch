from sklearn.datasets import load_boston as boston
from sklearn.neural_network import MLPRegressor
from utils import *


def load_boston():
    data_set = boston()
    X = data_set.data
    y = data_set.target.reshape(-1, 1)
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    y = MinMaxScaler((-1, 1)).fit_transform(y)
    return X, y.flatten()


X, y = load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPRegressor([100], activation='relu', alpha=0.01, solver='lbfgs')
mlp.fit(X_train, y_train)

acc_train = mlp.score(X_train, y_train)
acc_test = mlp.score(X_test, y_test)

print('train acc', acc_train)
print('test acc', acc_test)
