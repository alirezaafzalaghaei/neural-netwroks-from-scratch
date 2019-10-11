# author : Alireza Afzal Aghaei
# references: https://matrices.io/deep-neural-network-from-scratch/
#             https://github.com/theflofly/dnn_from_scratch_py


import numpy as np

np.random.seed(1)


class MLP:
    def __init__(self, hidden_layer_sizes: list,
                 epochs: int = 1000,
                 eta: float = 0.01,
                 beta: float = 1,
                 alpha: float = 0.01,
                 mu: float = 0.9,  # todo
                 ):
        self.random = np.random.randn
        self.weights_ = []
        self.beta = beta
        self.alpha = alpha
        self.n_epochs = epochs
        self.learning_rate = eta
        self.momentum = mu
        self.x = None
        self.y = None
        self.a = []
        self.z = []
        self.deltas = []
        self.dLdws = []
        self.hidden_layer_sizes = hidden_layer_sizes

    def init_weights_(self):
        for a, b in zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            self.weights_.append(self.random(a + 1, b) * self.beta)

    def _forward(self, a):
        self.z = []
        self.a = [a]
        for i in range(len(self.weights_)):
            z = a @ self.weights_[i]
            a = self.activation(z)
            if i != len(self.weights_) - 1:
                a = np.hstack((a, np.ones((a.shape[0], 1))))
            self.a.append(a)
            self.z.append(z)

        return self.a[-1]

    def _backward(self):
        self.deltas = list()
        self.dLdws = list()

        self.deltas.append((self.loss_prime(self.y, self.a[-1])) * self.activation_prime(self.z[-1]))
        self.dLdws.append(self.a[-2].T @ self.deltas[-1] + self.alpha * self.weights_[-1])

        for i in range(len(self.z) - 1, 0, -1):
            t = np.hstack((self.z[i - 1], np.ones((self.z[i - 1].shape[0], 1))))
            t = self.activation_prime(t)
            delta = ((self.deltas[-1] @ self.weights_[i].T) * t)
            delta = delta[:, :-1]
            self.deltas.append(delta)
            dLdw = self.a[i - 1].T @ self.deltas[-1] + self.alpha * self.weights_[i - 1]
            self.dLdws.append(dLdw)

        self.dLdws = self.dLdws[::-1]

    def _update_weights(self):
        for i in range(len(self.weights_)):
            self.weights_[i] -= self.learning_rate * self.dLdws[i] * 1 / len(self.x)

    def cost(self, t, y):
        return self.loss(y, t) + self.alpha * np.sum([w.sum() for w in self.weights_])

    @staticmethod
    def loss(t, y):
        return .5 * np.sum((t - y) ** 2)

    @staticmethod
    def loss_prime(t, y):
        return y - t

    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def activation_prime(x):
        return 1 - np.tanh(x) ** 2

    def fit(self, x, y):
        self.hidden_layer_sizes.insert(0, x.shape[1])
        self.hidden_layer_sizes.append(y.shape[1])

        self.x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.y = y
        self.init_weights_()
        hist = []
        for i in range(self.n_epochs):
            yp = self._forward(self.x)
            self._backward()
            self._update_weights()
            hist.append(self.loss(yp, self.y))
        return hist

    def predict(self, x):
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        return self._forward(x)
