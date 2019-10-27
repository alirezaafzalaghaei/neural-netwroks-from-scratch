from utils import *


def load_airfoil():
    data_set = np.loadtxt('./datasets/airfoil.dat')
    X = data_set[:, :-1]
    y = data_set[:, -1].reshape(-1, 1)
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    y = MinMaxScaler((-1, 1)).fit_transform(y)
    return X, y


X, y = load_airfoil()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLP([15, 15, 15], activation=LeakyReLu, epochs=2000, beta=.2, eta=.2, mu=.95, alpha=0, verbose=100,task='regression')

hist = mlp.fit(X_train, y_train)
yte_p = mlp.predict(X_test)
ytr_p = mlp.predict(X_train)

print('train r2', r2_score(y_train, ytr_p))
print('test  r2', r2_score(y_test, yte_p))

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
