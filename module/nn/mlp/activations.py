import abc
import numpy as np


class Activation(abc.ABC):

    @abc.abstractmethod
    def activation(self, x):
        pass

    @abc.abstractmethod
    def activation_prime(self, x):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class Tanh(Activation):
    def activation(self, x):
        return np.tanh(x)

    def activation_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def __str__(self):
        return 'Tanh'


class ReLu(Activation):
    def activation(self, x):
        return np.maximum(0, x)

    def activation_prime(self, x):
        return np.where(x <= 0, 0, 1)

    def __str__(self):
        return 'ReLu'


class LeakyReLu(Activation):
    def __init__(self, leakage=0.01):
        self.leakage = leakage

    def activation(self, x):
        y = np.copy(x)
        y[y < 0] *= self.leakage
        return y

    def activation_prime(self, x):
        return np.clip(x > 0, self.leakage, 1.0)

    def __str__(self):
        return 'LeakyReLu(%g)' % self.leakage


class Sigmoid(Activation):
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_prime(self, x):
        y = self.activation(x)
        return y * (1 - y)

    def __str__(self):
        return 'Sigmoid'


class Identity(Activation):
    def activation(self, x):
        return x

    def activation_prime(self, x):
        return 1

    def __str__(self):
        return 'Identity'


class Softmax(Activation):
    def activation(self, x):
        exp = np.exp(x - x.max(axis=1,keepdims=True))
        a = exp / np.sum(exp, axis=1, keepdims=True)
        return a

    def activation_prime(self, x):
        return 1  # just for simplicity

    def __str__(self):
        return 'Softmax'
