from sklearn.neural_network import MLPClassifier
from utils import *


def load_iris():
    iris = pd.read_csv('./datasets/iris.csv')
    X = iris.iloc[:, :-1].to_numpy()
    y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

    X = StandardScaler().fit_transform(X)
    # y = OneHotEncoder().fit_transform(y).toarray()
    y = LabelEncoder().fit_transform(y).flatten()  # .reshape(-1,1)
    return X, y


X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPClassifier([5, 5, 5, 5], activation='tanh', alpha=0.01, solver='lbfgs')
mlp.fit(X_train, y_train)

acc_train = mlp.score(X_train, y_train)
acc_test = mlp.score(X_test, y_test)

print('train acc', acc_train)
print('test acc', acc_test)
