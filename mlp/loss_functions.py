import abc
import numpy as np


class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def loss(y, t):
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_prime(y, t):
        pass


class MSE(Loss):
    @staticmethod
    def loss(y, t):
        return .5 * np.sum((t - y) ** 2)

    @staticmethod
    def loss_prime(y, t):
        return t - y


class XEntropy(Loss):
    @staticmethod
    def loss(t, y):
        indices = np.argmax(t, axis=1).astype(int)
        probability = y[np.arange(len(y)), indices]
        log = np.log(probability)
        loss = -1.0 * np.sum(log) / len(log)
        return loss

    @staticmethod
    def loss_prime(y, t):
        return (t - y) / y.shape[0]
