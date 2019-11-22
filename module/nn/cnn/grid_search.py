import itertools
from pathos.multiprocessing import ProcessPool as Pool
from . import CNN
import csv
import numpy as np
import pandas as pd
from keras.layers import *


def CNN_model(architecture, epochs, batch_size, optimizer, task, x, y, xt, yt, loss, metrics):
    print('.', end='', flush=True)
    model = CNN(architecture, epochs, batch_size, optimizer, loss, metrics, task, verbose=0)
    h = model.fit(x, y)
    sc = model.score(xt, yt)[1]
    t1,t2 = np.array(h.history['loss']), np.array(h.history['accuracy']).flatten()
    hist = np.empty((t1.shape[0],2))
    hist[:,0] = t1
    hist[:,1] = t2
    return hist, sc


class CNNGridSearch:
    def __init__(self, architecture, epochs, batch_sizes, optimizers, file_name='result.csv'):
        self.task = 'classification'
        if self.task == 'classification':
            self.loss_function = ''
        elif self.task == 'regression':
            self.loss_function = 'mean_squared_error'
        self.architecture = architecture
        self.epochs = epochs
        self.batch_sizes = batch_sizes
        self.optimizers = optimizers
        self.metrics = ['accuracy']
        self.csv = file_name

    def run(self, x, y, xt, yt):
        if self.task == 'classification':
            if y.shape[1] == 2:
                self.loss_function = 'binary_crossentropy'
            elif y.shape[1] > 2:
                self.loss_function = 'categorical_crossentropy'

        file = open(self.csv, 'w')
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['architecture', 'epochs', 'batch_size', 'optimizer', 'test_score', 'history_loss', 'history_score'])
        combinations = tuple(
            itertools.product(self.architecture, self.epochs, self.batch_sizes, self.optimizers))
        print("number of different models = %d" % len(combinations))
        print('.' * len(combinations))
        histories = list(map(  # Pool(1).
            lambda p: CNN_model(*p, self.task, x, y, xt, yt, self.loss_function, self.metrics),
            combinations))
        for (hist, sc), cfgs in zip(histories, combinations):
            h = list(hist[:, 0])
            hs = list(hist[:, 1])
            arch = cfgs[0]
            res = []
            for a in arch:
                _c = a.get_config()
                _c['_type'] = type(a)
                res.append(_c)
            opt = cfgs[-1].get_config()
            opt['_type'] = type(cfgs[-1])
            writer.writerow([res, *cfgs[1:-1], opt, sc, h, hs])
        file.close()
        print()
        return histories

    def best_model(self):
        df = pd.read_csv(self.csv)
        df['history_loss'] = df['history_loss'].apply(lambda x:
                                                      np.fromstring(
                                                          x.replace('\n', '').replace('[', '').replace(']',
                                                                                                       '').replace(
                                                              '  ',
                                                              ' '),
                                                          sep=','))
        # losses = np.array([np.abs(loss[-5:]).mean() for loss in df['history']])
        # index = losses.argmin()

        index = df['test_score'].to_numpy().argmax()
        return dict(df.iloc[index, :])
