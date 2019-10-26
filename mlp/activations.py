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


class Sigmoid(Activation):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_prime(x):
        y = Sigmoid.activation(x)
        return y * (1 - y)