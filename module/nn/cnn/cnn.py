from keras.models import Sequential
from keras.layers import InputLayer, Dense


class CNN:
    def __init__(self, architecture, epochs, batch_size, optimizer, loss, metrics, task, verbose=2):
        self.model = None
        self.batch_size = batch_size
        self.architecture = list(architecture)
        self.optimizer = optimizer
        self.task = task
        self.loss = loss
        self.verbose = verbose
        self.metrics = metrics
        self.epochs = epochs

    def init_model(self):
        self.model = Sequential(self.architecture)

    def fit(self, x_train, y_train, **kwargs):
        self.architecture.insert(0, InputLayer(input_shape=tuple(x_train.shape[1:])))
        if self.task == 'classification':
            self.architecture.append(Dense(y_train.shape[1], activation='softmax'))
        elif self.task == 'regression':
            self.architecture.append(Dense(y_train.shape[1]))
        self.init_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                              **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.evaluate(x, y, batch_size=self.batch_size, verbose=self.verbose)
