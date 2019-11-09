# author : Alireza Afzal Aghaei
# references: https://matrices.io/deep-neural-network-from-scratch/
#             https://github.com/theflofly/dnn_from_scratch_py


from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from .activations import *
from .loss_functions import *

np.random.seed(1)
np.seterr(all='raise')


class MLP:
    def __init__(self, hidden_layer_sizes: list,
                 activation: Activation = Tanh(),
                 epochs: int = 1000,
                 eta: float = 0.01,
                 beta: float = 1,
                 alpha: float = 0.01,
                 mu: float = 0.9,
                 batch_size: int = 128,
                 verbose: int = False,
                 task: str = ''
                 ):
        self._validation = tuple()
        self.random = np.random.randn
        self.weights_ = []
        self.beta = beta
        self.alpha = alpha
        self.n_epochs = epochs
        self.learning_rate = eta
        self.momentum = mu
        self.x = None
        self.y = None
        self.a = []
        self.z = []
        self.deltas = []
        self.dLdws = []
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.verbose = verbose
        self.activation = activation.activation
        self.activation_prime = activation.activation_prime
        self.current_loss = None
        self.velocity_ = []
        self.task = task
        self.batch_size = batch_size
        if task == 'classification':
            self._output = Softmax().activation
            self._output_prime = Softmax().activation_prime
            self._loss = XEntropy.loss
            self._loss_prime = XEntropy.loss_prime

        elif task == 'regression':
            self._output = Identity().activation
            self._output_prime = Identity().activation_prime
            self._loss = MSE.loss
            self._loss_prime = MSE.loss_prime
        else:
            raise ValueError('just use classification or regression')

    def init_weights(self):
        for a, b in zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            self.weights_.append(self.random(a + 1, b) * self.beta)
            self.velocity_.append(np.zeros((a + 1, b)))

    def _forward(self, a):
        self.z = []
        self.a = [a]
        for i in range(len(self.weights_)):
            z = a @ self.weights_[i]
            if i != len(self.weights_) - 1:
                a = self.activation(z)
            else:
                a = self._output(z)

            if i != len(self.weights_) - 1:
                a = np.hstack((a, np.ones((a.shape[0], 1))))
            self.a.append(a)
            self.z.append(z)

        return self.a[-1]

    def _backward(self, y):
        self.deltas = list()
        self.dLdws = list()

        self.deltas.append(self._loss_prime(self.a[-1], y) * self._output_prime(self.z[-1]))
        self.dLdws.append(self.a[-2].T @ self.deltas[-1] + self.alpha * self.weights_[-1])
        for i in range(len(self.z) - 1, 0, -1):
            t = np.hstack((self.z[i - 1], np.ones((self.z[i - 1].shape[0], 1))))
            t = self.activation_prime(t)
            delta = ((self.deltas[-1] @ self.weights_[i].T) * t)[:, :-1]
            self.deltas.append(delta)
            dLdw = self.a[i - 1].T @ self.deltas[-1] + self.alpha * self.weights_[i - 1]
            self.dLdws.append(dLdw)

        self.dLdws = self.dLdws[::-1]

    def _update_weights(self):
        for i in range(len(self.weights_)):
            self.velocity_[i] = self.momentum * self.velocity_[i] + self.learning_rate * (
                        self.dLdws[i] / self.batch_size - self.alpha * self.weights_[i])
            self.weights_[i] -= self.velocity_[i]

    def cost(self, x, y):
        # x and y must be encoded
        c = 0
        avg = 0
        for i in range(0, x.shape[0], self.batch_size):
            batch_x = x[i:min(i + self.batch_size, x.shape[0]), :]
            batch_y = y[i:min(i + self.batch_size, y.shape[0]), :]
            yp = self._forward(batch_x)
            avg += self._loss(yp, batch_y)
            c += 1
        return avg / c + self.alpha * np.sum([np.linalg.norm(w) for w in self.weights_])

    def fit(self, x_train, y_train, validation: list = False):
        x, y = self._encoder(x_train, y_train, fit=True)
        self.batch_size = min([self.batch_size, len(x)])
        self.hidden_layer_sizes.insert(0, x.shape[1])
        self.hidden_layer_sizes.append(y.shape[1])

        self.x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.y = y

        self.init_weights()
        hist = []
        valid = []
        sc = self.score(x_train, y_train)
        cost = self.cost(self.x, self.y)
        hist.append((cost, sc))
        self.current_loss = hist[-1]

        if validation:
            validation = list(validation)
            valid_x, valid_y = self._encoder(*validation)
            valid_x = np.hstack((valid_x, np.ones((valid_x.shape[0], 1))))
            print('epoch, train loss, train accuracy, validation loss, validation accuracy')
            _c = self.cost(valid_x, valid_y)
            s = self.score(*validation)
            valid.append((_c, s))
            self._validation = valid[-1]
        self._verbose(-1)
        for i in range(self.n_epochs):
            self.x, self.y = shuffle(self.x, self.y)
            for index in range(0, self.x.shape[0], self.batch_size):
                batch_x = self.x[index:min(index + self.batch_size, self.x.shape[0]), :]
                batch_y = self.y[index:min(index + self.batch_size, self.y.shape[0]), :]
                self._forward(batch_x)
                self._backward(batch_y)
                self._update_weights()
            if validation:
                _c, s = self.cost(valid_x, valid_y), self.score(*validation)
                valid.append((_c, s))
                self._validation = valid[-1]
            sc = self.score(x_train, y_train)
            _c = self.cost(self.x, self.y)
            hist.append((_c, sc))
            self.current_loss = hist[-1]
            self._verbose(i)
        if validation:
            return np.array(hist), np.array(valid)
        return np.array(hist)

    def _verbose(self, i):
        if self.verbose and (i + 1) % self.verbose == 0:
            if self._validation:
                # print("epoch %03d, train loss: %.6f, train score: %.6f, validation loss: %.6f, validation score: %.6f"
                #  % (i + 1, *self.current_loss, *self._validation))
                print("%04d, %05.8e, %.4f, %05.8e, %.4f" % (i + 1, *self.current_loss, *self._validation))
            else:
                print("epoch %05d, loss: %06.2f" % (i + 1, self.current_loss[0]))

    def predict(self, x):
        x = self._encoder(x)
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        y = self._forward(x)
        y = self._decoder(y)
        return y

    def score(self, x, y):
        yp = self.predict(x)
        # yp = self._encoder(y=yp)
        # y = self._encoder(y=y)
        try:
            if self.task == 'regression':
                return r2_score(y, yp)
            else:
                return accuracy_score(y, yp)
        except:
            return 0

    def _encoder(self, x=None, y=None, fit=False):
        if fit:
            self.x_encoder = StandardScaler().fit(x)
            if self.task == 'regression':
                self.y_encoder = StandardScaler().fit(y)
            else:
                self.y_encoder = OneHotEncoder(categories='auto').fit(y)

        if x is None:
            y = self.y_encoder.transform(y)
            return y.toarray() if self.task == 'classification' else y
        if y is None:
            return self.x_encoder.transform(x)
        y = self.y_encoder.transform(y)
        return self.x_encoder.transform(x), y.toarray() if self.task == 'classification' else y

    def _decoder(self, y):
        if self.task == 'classification':
            y = y.argmax(axis=1)
            y = np.eye(self.y.shape[1])[y]
        return self.y_encoder.inverse_transform(y)
