import numpy as np
import pandas as pd
from mlp import MLP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set()


def accuracy(y, t, eps=0.05):
    c = t - y
    c = c[np.abs(c) > eps]
    acc = 100 * (1 - c.shape[0] / y.shape[0])
    return acc
