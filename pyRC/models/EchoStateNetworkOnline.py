import numpy as np

from .BaseModel import OnlineReservoirModel
from .EchoStateNetwork import parallelReservoirESN, EchoStateNetwork, parallelESN
from . import util
from scipy.stats import multivariate_normal, norm
import tqdm
from matplotlib import pyplot as plt


class OnlineESN(OnlineReservoirModel, EchoStateNetwork):

    def train(self, feature, target=None, verbose=False):
        if target is None:
            feature, target = feature[:-1, :], feature[1:, :]

        if feature.ndim == 2:
            if verbose:
                with tqdm.tqdm(total=feature.shape[0]) as bar:

                    def helper(z):
                        x, y = z[:-1], z[-1:]
                        bar.update()
                        return self.train(x, y)

                    return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))
            else:
                def helper(z):
                    x, y = z[:-1], z[-1:]
                    return self.train(x, y)

                return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))

        return self._train_and_predict(feature, target)


# class parallelOnlineESN(OnlineReservoirModel, parallelReservoirESN):
#     pass

class parallelOnlineESN(parallelESN):
    _network_class = OnlineESN

    def train(self, feature, target=None, verbose=False):
        if target is None:
            feature, target = feature[:-1, :], feature[1:, :]

        if feature.ndim == 2:
            if verbose:
                with tqdm.tqdm(total=feature.shape[0]) as bar:

                    def helper(z):
                        x, y = z[:-1], z[-1:]
                        bar.update()
                        return self.train(x, y)

                    return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))
            else:
                def helper(z):
                    x, y = z[:-1], z[-1:]
                    bar.update()
                    return self.train(x, y)

                return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))

        ret = np.hstack([n._train_and_predict(feature, target) for n in self.networks])
        return self.average(ret)


class OnlineExpertESN(parallelOnlineESN):

    def __init__(self, *args, learning_rate=1e-2, error_fun=None, lookback=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.stored_weights = np.ones((1, self.num), dtype=np.float64) / self.num
        if error_fun is None:
            error_fun = util.MSE
        self.error_fun = error_fun
        # self.lookback = lookback
        # if lookback > 1:
        #     self.errors = np.zeros((lookback, self.num), dtype=np.float64)
        # else:
        #     self.errors = None

    def train(self, feature, target=None, verbose=False):
        if target is None:
            feature, target = feature[:-1, :], feature[1:, :]

        if feature.ndim == 2:
            if verbose:
                with tqdm.tqdm(total=feature.shape[0]) as bar:

                    def helper(z):
                        x, y = z[:-1], z[-1:]
                        bar.update()
                        return self.train(x, y)

                    return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))
            else:
                def helper(z):
                    x, y = z[:-1], z[-1:]
                    return self.train(x, y)

                return np.apply_along_axis(helper, axis=1, arr=np.hstack([feature, target]))

        fitted = np.hstack([n._train_and_predict(feature, target) for n in self.networks])
        errors = np.apply_along_axis(lambda x: self.error_fun(x, target.flatten()), axis=0, arr=fitted)
        self._update_weights(errors.flatten())
        return self.average(fitted.flatten())

    def _update_weights(self, errors):
        if errors.shape[0] != self.num:
            raise ValueError

        if np.max(errors) != np.min(errors):
            errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))

        # if np.sum(errors) == 0.0:
        #     return

        # if self.lookback > 1:
        #     errors = self.append_errors(errors)

        self.weights *= np.exp(-self.learning_rate*errors)
        self.weights /= np.sum(self.weights)
        self.weights = np.maximum(self.weights, 0.0001)
        self.weights /= np.sum(self.weights)
        self.stored_weights = np.vstack((self.stored_weights, self.weights.reshape((1, -1))))

    # def append_errors(self, errors):
    #     if self.lookback > 1:
    #         self.errors[:-1, :] = self.errors[1:, :]
    #         self.errors[-1, :] = errors.flatten()
    #     else:
    #         self.errors = errors
    #     return np.mean(self.errors, axis=0)


class OnlineUnivariateGaussianExpert(parallelOnlineESN):

    def __init__(self, network_class=None, **kwargs):
        super().__init__(**kwargs)

        for i, n in enumerate(self.networks):
            n.reservoir.update = self.save_update_wrapper(n.reservoir.update, i)

        self.write_log = True
        self.log_likelihood = np.ones((self.num, ), dtype=np.float64) / self.num
        self.stored_weights = np.ones((1, self.num), dtype=np.float64) / self.num

    def save_update_wrapper(self, network_update, i):
        def process_reservoir_update(array):
            x = network_update(array)

            if self.update_weights:
                self.log_likelihood[i] = util.univariate_likelihood(x, sigma2=self.networks[i].reservoir.activation.sigma2, log=True)

                if i == self.num - 1 and self.write_log:
                    # self.log_likelihood -= np.max(self.log_likelihood)
                    self.weights *= np.exp(self.scale(self.log_likelihood))
                    # if np.mean(self.weights) == 0.0:
                    #     self.weights =
                    # self.weights[self.weights == np.nan] = 0.0001
                    # self.weights[self.weights == np.inf] = 100
                    # self.weights /= self.num*np.mean(self.weights)
                    self.weights = np.maximum(self.weights, 0.0001)
                    self.weights /= np.sum(self.weights)
                    self.stored_weights = np.vstack((self.stored_weights, self.weights.reshape((1, -1))))
                    # self.stored_weights.append(self.weights.copy().flatten())

            return x
        return process_reservoir_update

    @staticmethod
    def scale(x):
        # return x
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def _wrap(self):
        for i, n in enumerate(self.networks):
            n.reservoir.update = self.save_update_wrapper(n.reservoir.update, i)
