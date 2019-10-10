import matplotlib.pyplot as plt
import numpy as np

from mlp import MLP

X = np.array([[1.4, -1, 0.4],
              [0.4, -1, 0.1],
              [5.4, -1, 4],
              [1.5, -1, 1],
              [1.8, 1, 1]])
y = np.array([[.45],
              [.8],
              [.2],
              [.5],
              [.55]
              ])

mlp = MLP([1], epochs=10000, beta=0.3, eta=1)
hist = mlp.run(X, y)
print(mlp.predict(X).round(12))
plt.plot(list(range(len(hist))), np.log(hist))
plt.title("%.2e" % hist[-1])
plt.show()
