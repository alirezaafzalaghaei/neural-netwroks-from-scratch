# author : Alireza Afzal Aghaei
# references: https://matrices.io/deep-neural-network-from-scratch/


import numpy as np
np.random.seed(1)

class MLP:
    def __init__(self, hidden_layer_sizes: list,
                 epochs: int = 1000,
                 eta: float = 0.01,
                 beta: float = 1,
                 alpha: float = 0.01  # todo
                 ):
        self.random = np.random.randn
        self.weights = []
        self.beta = beta
        self.alpha = alpha  # todo
        self.n_epochs = epochs
        self.learning_rate = eta
        self.x = None
        self.y = None
        self.a = []
        self.z = []
        self.deltas = []
        self.dLdws = []
        self.hidden_layer_sizes = hidden_layer_sizes

    def init_weights(self):
        for a, b in zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            self.weights.append(self.random(a + 1, b) * self.beta)

    def forward(self, a):

        self.z = []
        self.a = [a]
        for i in range(len(self.weights)):
            z = a @ self.weights[i]
            a = self.activation(z)
            if i != len(self.weights) - 1:
                a = np.hstack((a, np.ones((a.shape[0], 1))))
            self.a.append(a)
            self.z.append(z)

        return self.a[-1]

    def backward(self):
        self.deltas = list()
        self.dLdws = list()
        self.deltas.append((self.y - self.a[-1]) * -self.activation_prime(self.z[-1]))
        self.dLdws.append(self.a[-2].T @ self.deltas[-1])

        for i in range(len(self.z) - 1, 0, -1):
            t = np.hstack((self.z[i - 1], np.ones((self.z[i - 1].shape[0], 1))))
            t = self.activation_prime(t)
            delta = ((self.deltas[-1] @ self.weights[i].T) * t)
            delta = delta[:, :-1]
            self.deltas.append(delta)
            dLdw = self.a[i - 1].T @ self.deltas[-1]
            self.dLdws.append(dLdw)

        self.dLdws = self.dLdws[::-1]

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.dLdws[i] * 1 / len(self.x)

    @staticmethod
    def loss(y, t):
        return .5 * np.sum((t - y) ** 2)

    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def activation_prime(x):
        return 1 - np.tanh(x) ** 2

    def run(self, x, y):
        self.hidden_layer_sizes.insert(0, x.shape[1])
        self.hidden_layer_sizes.append(y.shape[1])

        self.x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.y = y
        self.init_weights()
        hist = []
        for i in range(self.n_epochs):
            yp = self.forward(self.x)
            self.backward()
            self.update_weights()
            hist.append(self.loss(yp, self.y))
        return hist

    def predict(self, x):
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        return self.forward(x)
