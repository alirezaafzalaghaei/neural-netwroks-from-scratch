from utils import *


def load_airfoil():
    data_set = np.loadtxt('./datasets/airfoil.dat')
    X = data_set[:, :-1]
    y = data_set[:, -1].reshape(-1, 1)
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(y)

    return X, y


X, y = load_airfoil()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLP([10, 10, 10, 10], activation=LeakyReLu(.01), epochs=200, batch_size=32, beta=.1, eta=.4, mu=.95, alpha=0,
          verbose=100, task='regression')

hist = mlp.fit(X_train, y_train)


print('train r2: %.2f %%' % (100 * mlp.score(X_train, y_train)))
print('test  r2: %.2f %%' % (100 * mlp.score(X_test, y_test)))

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
