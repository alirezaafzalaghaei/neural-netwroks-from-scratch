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

mlp = MLP([15, 15, 15], activation=ReLu, epochs=300, beta=.1, eta=.001,mu=.9, alpha=0, verbose=1)

hist = mlp.fit(X_train, y_train)
yte_p = mlp.predict(X_test)


loss_test = mlp.cost(y_test, yte_p)

print('train loss: %.4f' % hist[-1])
print('test  loss: %.4f' % loss_test)

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
