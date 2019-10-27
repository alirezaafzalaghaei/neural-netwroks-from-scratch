from sklearn.datasets import load_boston as boston
from utils import *


def load_boston():
    data_set = boston()
    X = data_set.data
    y = data_set.target
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    y = MinMaxScaler((-1, 1)).fit_transform(y.reshape(-1,1))
    return X, y


X, y = load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLP([100], activation=ReLu, epochs=10000, beta=0.1, eta=.06, mu=.95, alpha=0.02, verbose=100,type='regression')

hist = mlp.fit(X_train, y_train)
yte_p = mlp.predict(X_test)
ytr_p = mlp.predict(X_train)

print('train r2', r2_score(y_train,ytr_p))
print('test  r2', r2_score(y_test,yte_p))

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
