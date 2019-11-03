# author : Alireza Afzal Aghaei
# references: https://matrices.io/deep-neural-network-from-scratch/
#             https://github.com/theflofly/dnn_from_scratch_py


from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
        self._validation = False
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
            self.velocity_[i] = self.momentum * self.velocity_[i] + self.learning_rate * self.dLdws[i] / len(self.x)
            self.weights_[i] -= self.velocity_[i]

    def cost(self, t, y):
        return self._loss(y, t) + self.alpha * np.sum([w.sum() for w in self.weights_])

    def fit(self, x, y, validation: tuple=False):
        x, y = self._encoder(x, y, fit=True)

        self.hidden_layer_sizes.insert(0, x.shape[1])
        self.hidden_layer_sizes.append(y.shape[1])

        self.x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.y = y
        self.init_weights()
        hist = []
        valid = []
        for i in range(self.n_epochs):
            avg_cost = 0
            c = 0
            for index in range(0, self.x.shape[0], self.batch_size):
                batch_x = self.x[index:min(index + self.batch_size, self.x.shape[0]), :]
                batch_y = self.y[index:min(index + self.batch_size, self.y.shape[0]), :]
                try:
                    yp = self._forward(batch_x)
                    self._backward(batch_y)
                    self._update_weights()
                except Exception as ex:
                    return [str(ex)]
                avg_cost += self.cost(yp, batch_y)
                c += 1
            if validation:
                #valid.append(self.score(validation[0], validation[1]))
                yp = self.predict(validation[0])
                valid.append(self.cost(validation[1], yp))
                self._validation = valid[-1]

            # validation?
            hist.append(avg_cost / c)
            self.current_loss = hist[-1]
            self._verbose(i)
        if validation:
            return hist, valid
        return hist

    def _verbose(self, i):
        if self.verbose and (i + 1) % self.verbose == 0:
            if self._validation:
                print("epoch %05d, loss: %06.2f, validation: %.2f" % (i + 1, self.current_loss, self._validation))
            else:
                print("epoch %05d, loss: %06.2f" % (i + 1, self.current_loss))

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
