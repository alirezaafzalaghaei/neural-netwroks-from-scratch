from utils import *


def load_iris():
    iris = pd.read_csv('./datasets/iris.csv')
    X = iris.iloc[:, :-1].to_numpy()
    y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

    X = StandardScaler().fit_transform(X)
    y = OneHotEncoder().fit_transform(y).toarray()

    return X, y


X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLP([5, 5, 5, 5], activation=Tanh, epochs=5000, beta=1, eta=.5, alpha=.01)

hist = mlp.fit(X_train, y_train)
ytr_p = mlp.predict(X_train)
yte_p = mlp.predict(X_test)

acc_test = accuracy_score(y_test.argmax(axis=1), yte_p.argmax(axis=1))
acc_train = accuracy_score(y_train.argmax(axis=1), ytr_p.argmax(axis=1))

loss_test = mlp.loss(y_test, yte_p)

print('train loss: %.4f, accuracy: %.2f%%' % (hist[-1], acc_train * 100))
print('test  loss: %.4f, accuracy: %.2f%%' % (loss_test, acc_test * 100))

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()
