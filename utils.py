import numpy as np
import pandas as pd
from mlp.mlp import MLP, ReLu, Tanh, LeakyReLu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# disable pycharm's import optimizer
pd, MLP, plt, StandardScaler, MinMaxScaler, ReLu, Tanh, OneHotEncoder
train_test_split, accuracy_score, LabelEncoder,LeakyReLu, r2_score

sns.set()
