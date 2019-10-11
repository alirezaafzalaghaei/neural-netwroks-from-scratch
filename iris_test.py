import numpy as np
import pandas as pd
from mlp import MLP
import seaborn as sns
from utils import accuracy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set()

iris = pd.read_csv('./datasets/iris.csv')
X = iris.iloc[:, :-1].to_numpy()
y = iris.iloc[:, -1].to_numpy().reshape(-1, 1)

X = StandardScaler().fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

mlp = MLP([5, 5, 5, 5], epochs=5000, beta=1, eta=.5)
hist = mlp.run(Xtrain, ytrain)
ytr_p = mlp.predict(Xtrain)
yte_p = mlp.predict(Xtest)

acc_te = accuracy(ytest,yte_p)
acc_tr = accuracy(ytrain,ytr_p)

print('train loss: %.4f, accuracy: %.2f%%' % (hist[-1],acc_tr))
print('train loss: %.4f, accuracy: %.2f%%' % (mlp.loss(ytest, yte_p),acc_te))

plt.plot(list(range(len(hist))), np.log(hist))
plt.title("%.2e" % hist[-1])
plt.show()
