
from copy import copy
import numpy as np
from scipy.stats import multivariate_normal
from .BaseModel import ReservoirModel, OnlineReservoirModel
from .reservoirs import ESNReservoir, LeakyESNReservoir, DeepESNReservoir, ESNReservoirArray
from .output_models_online import WRLS
from ..util.scaler import tanh, identity
from .. import util


class EchoStateNetwork(ReservoirModel):

    _reservoir_class = LeakyESNReservoir

    def __repr__(self):
        return "({}: N={}, SR={})".format(self.__class__.__name__, self.size, self.spectral_radius)


class deepESN(EchoStateNetwork):
    _reservoir_class = DeepESNReservoir

    def add_layer(self, *args, **kwargs):
        self.reservoir.add_layer(*args, **kwargs)

    @property
    def layers(self):
        return getattr(self.reservoir, 'layers', None)


class parallelReservoirESN(EchoStateNetwork):
    _reservoir_class = ESNReservoirArray

    def add_reservoir(self, *args, **kwargs):
        self.reservoir.add_reservoir(*args, **kwargs)

    @property
    def reservoirs(self):
        return getattr(self.reservoir, 'reservoirs', None)


class parallelESN:

    _network_class = EchoStateNetwork

    def __init__(self, network_class=None, update_weights=True, **kwargs):
        if network_class is not None:
            self._rc_class = network_class
        self.networks = []
        self.add_network(**kwargs)
        self.num = len(self.networks)
        self.weights = np.ones((len(self.networks,)))
        self.update_weights = update_weights

    @util.make_kwargs_one_length
    def add_network(self, **kwargs):
        new_network = self._network_class(**kwargs)
        self.networks.append(new_network)

    def update(self, x):
        if x.ndim == 2:
            return np.apply_along_axis(self.update, arr=x, axis=1)

        for n in self.networks:
            n.update(x)

    def train(self, *args, **kwargs):
        for n in self.networks:
            n.train(*args, **kwargs)
        fitted = np.hstack([n.fitted for n in self.networks])
        return self.average(fitted)

    def average(self, x, axis=1):
        if isinstance(x, np.ndarray):
            if x.ndim == 1 or x.shape[0] == self.num and x.shape[1] != self.num:
                axis = 0
        return np.average(x, axis=axis, weights=self.weights)

    def predict(self, *args, **kwargs):
        pred = []
        for n in self.networks:
            pred.append(n.predict(*args, **kwargs))

        if pred[0].ndim == 1:
            pred = np.vstack(pred)
        elif pred[0].shape[0] == 1 and pred[0].shape[1] != 1:
            pred = np.vstack(pred).transpose()
        else:
            pred = np.hstack(pred)

        return self.average(pred)


class ExpertESN(parallelESN):

    def __init__(self, *args, learning_rate=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.stored_weights = []

    def train(self, *args, **kwargs):
        fitted = super().train(*args, **kwargs)
        errors = np.vstack([n._errors for n in self.networks]).reshape((-1, self.num))
        np.apply_along_axis(self._update_weights, axis=1, arr=errors)
        return fitted

    def _update_weights(self, errors):
        self.weights *= np.exp(-self.learning_rate*errors)
        self.weights /= np.sum(self.weights)
        self.stored_weights.append(self.weights.copy())


class UnivariateGaussianExpert(parallelESN):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        for i, n in enumerate(self.networks):
            n.reservoir.update = self.save_update_wrapper(n.reservoir.update, i)

        self.weights = np.ones((self.num, ), dtype=np.float)
        self.temp_weights = np.zeros((self.num,), dtype=np.float64)
        self.posterior = self.weights

        self.write_log = False
        self.stored_weights = []

    def save_update_wrapper(self, reservoir_update, i):
        def process_reservoir_update(array):
            if array.ndim == 2:
                return reservoir_update(array)

            x = reservoir_update(array)
            self.temp_weights[i] = util.univariate_likelihood(x, sigma2=self.networks[i].reservoir.activation.sigma2)

            if i == self.num - 1 and self.write_log:
                self.weights *= self.temp_weights
                self.weights /= np.sum(self.weights)
                self.stored_weights.append(self.weights.copy().flatten())

            return x
        return process_reservoir_update


class MultivariateGaussianMixtureESN(parallelESN):

    def __init__(self, *args, sigma_ini=0.01, **kwargs):
        super().__init__(**kwargs)

        self.sp = np.ones((self.num, ), dtype=np.float64)
        self.omega = np.ones((self.num, ), dtype=np.float64)

        self.mu = np.zeros((self.num, self.networks[0].size), dtype=np.float64)
        # self.C = np.stack(list(np.diag(np.ones((r.size,), dtype=np.float64)*sigma_ini) for r in self.networks), axis=0)
        self.C = np.stack(list(np.diag(np.ones((r.size,), dtype=np.float64)*r.reservoir.activation.sigma2) for r in self.networks), axis=0)

        self.ll = np.ones((self.num, ), dtype=np.float64)
        self.prior = np.ones((self.num, ), dtype=np.float64)
        self.posterior = np.ones((self.num, ), dtype=np.float64)

        for i, n in enumerate(self.networks):
            n.reservoir.update = self.save_update_wrapper(n.reservoir.update, i)

        self.update_gm = False
        self.sigma_ini = sigma_ini

        self.weights = self.posterior

        self.stored_x = np.zeros((self.num, self.networks[0].size), dtype=np.float64)

        self.learning_rate = 0.95
        self.write_log = False
        self.stored_weights = []

    def reset_mu_sigma(self):
        self.mu = np.zeros((self.num, self.networks[0].size), dtype=np.float64)
        self.C = np.stack(list(np.diag(np.ones((r.size,), dtype=np.float64)*self.sigma_ini) for r in self.reservoirs), axis=0)

    def save_update_wrapper(self, reservoir_update, i):
        def process_reservoir_update(array):
            # if array.ndim == 2:
            #     return fun(array)
            array = array.flatten()
            x = reservoir_update(array)
            #self.stored_x[i, :] = x
            #c = np.diag(np.ones((self.networks[i].size,))*self.networks[i].reservoir.activation.sigma2)
            self.weights[i] = multivariate_normal.pdf(x=x.reshape((1, -1)), mean=self.mu[i, ...], cov=self.C[i, ...])

            if i != 0:
                bla = 1

            if i == self.num - 1 and self.update_gm and False:

                self.posterior[:] = self.ll * self.prior / np.sum(self.ll*self.prior)
                self.sp += self.posterior

                self.omega = self.posterior/self.sp

                # new_mu = self.mu + self.omega.reshape((-1, 1)) * (self.stored_x - self.mu)
                new_mu = self.mu + self.learning_rate * (self.stored_x - self.mu)

                x = self.stored_x.flatten()
                for j in range(len(self.networks)):
                    self.C[j, ...] = self.C[j, ...] + \
                        np.outer(new_mu[j, ...] - self.mu[j, ...], new_mu[j, ...] - self.mu[j, ...]) + \
                        self.omega[i] * (np.outer(self.stored_x[j, :] - self.mu[j, ...], self.stored_x[j, :] - self.mu[j, ...]) - self.C[j, ...])
                    # self.learning_rate * (np.outer(self.stored_x[j, :] - self.mu[j, ...], self.stored_x[j, :] - self.mu[j, ...]) - self.C[j, ...])

                    self.mu = new_mu
                self.prior = self.sp / np.sum(self.sp)

            if i == self.num - 1:
                self.weights /= np.sum(self.weights)

            if i == self.num - 1 and self.write_log:
                self.stored_weights.append(self.weights.copy().flatten())

            return x
        return process_reservoir_update


class ExpertGaussianMixtureESN(ExpertESN, UnivariateGaussianExpert):

    def __init__(self, *args, learning_rate=1e-2, **kwargs):
        self.learning_rate = learning_rate
        UnivariateGaussianMixtureESN.__init__(self, *args, **kwargs)

    def average(self, x, axis=1):
        if isinstance(x, np.ndarray):
            if x.ndim == 1 or x.shape[0] == self.num:
                axis = 0

        return np.average(x, axis=axis, weights=self.mixed_weights)

    @property
    def mixed_weights(self):
        r = self.posterior + self.weights
        return r/np.sum(r)
