from utils import *
import time

def load_iris():
    iris = pd.read_csv('../datasets/iris.csv')
    X = iris.iloc[:, :-1].to_numpy()
    y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

    X = StandardScaler().fit_transform(X)
    y = OneHotEncoder().fit_transform(y).toarray()

    return X, y


hidden_layers = [(5,)]
activations = [Tanh(), LeakyReLu(.1)]
batch_sizes = [32]
epochs = [20]
mus = [.85]
betas = [.1,]
etas = [.01]
alphas = [.001]

X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y)
t = time.time()
mlp = MLPGridSearch('classification', hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas)
histories = mlp.run(X_train, y_train)
t = time.time() - t
print(t)

result = mlp.best_model()
hist = result.pop('history')
for name,value in result.items():
    print("%-13s : %s" % (name,value))
plt.plot(list(range(len(hist))), hist)
plt.title("loss: %.2e" % hist[-1])
plt.xlabel('iterations')
plt.ylabel('Log(loss)')
plt.show()

