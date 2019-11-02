import itertools

import csv
import cupy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool

from . import MLP


def MLP_model(hidden_layer, activation, epoch, eta, beta, alpha, mu, batch_size, task, x, y, xt, yt):
    print('.', end='', flush=True)
    mlp = MLP(hidden_layer, activation, epoch, eta, beta, alpha, mu, batch_size, False, task)
    hist = mlp.fit(x, y)
    score = mlp.score(xt, yt)
    return hist, score


class MLPGridSearch:
    def __init__(self, task, hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas,
                 file_name='result.csv'):
        self.task = task
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.mus = mus
        self.betas = betas
        self.etas = etas
        self.alphas = alphas
        self.csv = file_name

    def run(self, x, y, xt, yt):
        file = open(self.csv, 'w')
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['hidden_layer', 'activation', 'epoch', 'eta', 'beta', 'alpha', 'mu', 'batch_size', 'test_score',
             'history'])
        combinations = tuple(
            itertools.product(self.hidden_layers, self.activations, self.epochs, self.etas, self.betas, self.alphas,
                              self.mus, self.batch_sizes))
        print("number of different models = %d" % len(combinations))
        print('.' * len(combinations))
        histories = Pool().map(lambda p: MLP_model(*p, self.task, x, y, xt, yt), combinations)
        for (hist, loss), cfg in zip(histories, combinations):
            writer.writerow([*cfg, loss, hist])
        file.close()
        print()
        return histories

    def best_model(self):
        df = pd.read_csv(self.csv)
        df['history'] = df['history'].apply(lambda x:
                                            np.fromstring(
                                                x.replace('\n', '').replace('[', '').replace(']', '').replace('  ',
                                                                                                              ' '),
                                                sep=','))
        # losses = np.array([np.abs(loss[-5:]).mean() for loss in df['history']])
        # index = losses.argmin()

        index = df['test_score'].to_numpy().argmax()
        return dict(df.iloc[index, :])

# todo: rank models
