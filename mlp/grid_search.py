import csv
from mlp.mlp import MLP
import itertools
from utils import pd, np
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Poool


def MLP_model(hidden_layer, activation, epoch, eta, beta, alpha, mu, batch_size, task, x, y):
    mlp = MLP(hidden_layer, activation, epoch, eta, beta, alpha, mu, batch_size, False, task)
    hist = mlp.fit(x, y)
    return hist


class MLPGridSearch:
    def __init__(self, task, hidden_layers, activations, batch_sizes, epochs, mus, betas, etas, alphas):
        self.task = task
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.mus = mus
        self.betas = betas
        self.etas = etas
        self.alphas = alphas
        self.csv = open('result.csv', 'w')
        self.writer = csv.writer(self.csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(['hidden_layer', 'activation', 'epoch', 'eta', 'beta', 'alpha', 'mu', 'batch_size','history'])

    def run(self, x, y):
        c = 0
        combinations = tuple(itertools.product(self.hidden_layers, self.activations, self.epochs, self.etas, self.betas, self.alphas, self.mus, self.batch_sizes))
        print("number of models = %d" % len(combinations))
        print('0000 - 0050', end=': ')

        histories = map(lambda p: MLP_model(*p, self.task, x, y), combinations)
        for hist, cfg in zip(histories, combinations):
            self.writer.writerow([*cfg, hist])
            c += 1
            print('.', end='')
            if c % 50 == 0:
                print('\n%04d - %04d' % (c, c + 50), end=': ')
        print()
        self.csv.close()
        return histories

    def best_model(self):
        df = pd.read_csv('result.csv')
        df['history'] = df['history'].apply(lambda x:
                                   np.fromstring(
                                       x.replace('\n','')
                                        .replace('[','')
                                        .replace(']','')
                                        .replace('  ',' '), sep=','))
        losses = np.array([np.abs(loss[-5:]).mean() for loss in df['history']])
        index = losses.argmin()
        return dict(df.iloc[index, :])

# todo: rank models
