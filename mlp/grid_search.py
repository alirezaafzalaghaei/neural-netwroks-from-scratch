import csv
from mlp.mlp import MLP
import itertools
from utils import pd, np
from pathos.multiprocessing import ProcessingPool as Pool


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
        self.csv = 'result.csv'

    def run(self, x, y):
        file = open(self.csv, 'w')
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['hidden_layer', 'activation', 'epoch', 'eta', 'beta', 'alpha', 'mu', 'batch_size', 'history'])
        c = 0
        combinations = tuple(
            itertools.product(self.hidden_layers, self.activations, self.epochs, self.etas, self.betas, self.alphas,
                              self.mus, self.batch_sizes))
        print("number of different models = %d" % len(combinations))
        histories = Pool().map(lambda p: MLP_model(*p, self.task, x, y), combinations)
        for hist, cfg in zip(histories, combinations):
            writer.writerow([*cfg, hist])
        file.close()
        return histories

    def best_model(self):
        df = pd.read_csv('result.csv')
        df['history'] = df['history'].apply(lambda x:
                                            np.fromstring(
                                                x.replace('\n', '').replace('[', '').replace(']', '').replace('  ',
                                                                                                              ' '),
                                                sep=','))
        losses = np.array([np.abs(loss[-5:]).mean() for loss in df['history']])
        index = losses.argmin()
        return dict(df.iloc[index, :])

# todo: rank models
