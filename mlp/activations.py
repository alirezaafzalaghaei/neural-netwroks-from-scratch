import abc
import numpy as np


class Activation(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def activation(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def activation_prime(x):
        pass


class Tanh(Activation):
    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def activation_prime(x):
        return 1 - np.tanh(x) ** 2


class ReLu(Activation):
    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def activation_prime(x):
        return np.where(x <= 0, 0, 1)


class LeakyReLu(Activation):
    @staticmethod
    def activation(x):
        return np.where(x > 0, x, x * 0.01)

    @staticmethod
    def activation_prime(x):
        dx = np.ones_like(x)
        dx[x < 0] = .1
        return dx


class Sigmoid(Activation):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_prime(x):
        y = Sigmoid.activation(x)
        return y * (1 - y)


class Identity(Activation):
    @staticmethod
    def activation(x):
        return x

    @staticmethod
    def activation_prime(x):
        return 1


class Softmax(Activation):
    @staticmethod
    def activation(x):
        exp = np.exp(x)
        a = exp / np.sum(exp, axis=1, keepdims=True)
        # print(x.shape, a.shape)
        return a

    @staticmethod
    def activation_prime(x):
        return 1 # just for simplicity
