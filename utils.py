import numpy as np
def accuracy(y, t, eps=0.05):
    c = t - y
    c = c[np.abs(c) > eps]
    acc = 100 * (1 - c.shape[0] / y.shape[0])
    return acc