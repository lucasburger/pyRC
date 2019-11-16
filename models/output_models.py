
import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifier, Lars
import sklearn.linear_model
from copy import deepcopy
import util
import numba

from abc import ABCMeta, abstractmethod, abstractproperty


class BaseOutputModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        """
        This function must be overridden by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        This function must be overridden by subclasses.
        """
        pass

    @abstractproperty
    def error(self):
        pass


class MinEigenvalueRegression(BaseOutputModel):

    _alphas = []

    def __init__(self, intercept=True, min_eigenvalue=None, ts_split=None, **kwargs):

        self.intercept = intercept
        self.min_eigenvalue = min_eigenvalue

        self.ts_split = ts_split

        self.size = None
        self.beta = None
        self.regularize_lambda = None
        self.fitted = None
        self._error = None

    def check_input(self, x):
        if self.size:
            assert x.shape[-1] == self.size

    def add_intercept(self, x):
        if not self.intercept:
            return x

        if x.ndim == 1:
            if self.size and (x[0] != 1 or x.shape[0] == self.size - 1):
                x = np.hstack((np.ones(shape=(1,)), x)).flatten()
        else:
            if x.shape[1] == 1 and self.size is None:
                x = np.hstack((np.ones((1,)), x.flatten()))
            elif np.any(x[:, 0].flatten() != 1):
                x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
            elif self.size and x.shape[0] == self.size - self.intercept:
                x = np.hstack((np.ones((1, 1)), x)).T

        return x

    def fit(self, x, y, alphas=None, error_fun=util.RMSE, **kwargs):
        self.size = None
        x = self.add_intercept(x)
        self.check_input(x)
        self.size = x.shape[-1]

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if self.ts_split is None:
            self.ts_split = util.ts_split(x.shape[0], test_set_size=0.1)

        errors = []
        for id_train, id_test in self.ts_split:
            x_train, y_train = x[id_train, :], y[id_train, :]
            x_test, y_test = x[id_test, :], y[id_test, :]

            xxt = np.dot(x_train.T, x_train)

            self.regularize_lambda = self.min_eigenvalue - min(np.linalg.eigvals(xxt))

            if isinstance(self.regularize_lambda, complex):
                self.regularize_lambda = np.real(self.regularize_lambda)

            beta = np.linalg.pinv(xxt + self.regularize_lambda * self.size * np.eye(self.size)) @ x_train.T @ y_train
            errors.append(util.RMSE(np.dot(x_test, beta), y_test))

        self.error = np.mean(errors)

        xx = np.dot(x.T, x)

        self.regularize_lambda = self.min_eigenvalue - min(np.linalg.eigvals(xx))

        if isinstance(self.regularize_lambda, complex):
            self.regularize_lambda = np.real(self.regularize_lambda)

        self.beta = np.linalg.pinv(xx + self.regularize_lambda * self.size * np.eye(self.size)) @ x.T @ y
        # self.fitted = np.dot(x, self.beta)
        # self.error = util.RMSE(self.fitted, y)

        return

    def predict(self, x):
        x = self.add_intercept(x)
        self.check_input(x)
        return np.dot(x, self.beta)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, other):
        self._error = other

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, other):
        self._size = other


class GaussianElimination:

    def __init__(self, add_intercept: bool = True, ridgelambda: float = 0.0):
        self.add_intercept = True
        self.beta = None
        self.ridgelambda = ridgelambda
        self.fitted = None
        self._error = None

    def fit(self, x: np.ndarray, y: np.ndarray):

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))

        if self.ridgelambda > 0.0:
            self.beta = np.linalg.pinv(np.dot(x.T, x) + self.ridgelambda * np.eye(x.shape[1])) @ x.T @ y
        else:
            self.beta = np.dot(np.linalg.pinv(x), y)

        self.fitted = np.dot(x, self.beta)
        self._error = util.NRMSE(y, self.fitted)
        return

    def predict(self, x: np.ndarray):
        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        return np.dot(x, self.beta)


spec = [
    ('add_intercept', numba.bool_),
    # ('ridgelambda', numba.float32),               # a simple scalar field
    # ('_error', numba.float32),               # a simple scalar field
    ('beta', numba.float64[:, :]),
    # ('fitted', numba.float64[:])          # an array field
]


@numba.jitclass(spec)
class NumbaGaussianElimination:

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = True
        self.beta = np.zeros(shape=(1, 1), dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray):

        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))

        self.beta = np.dot(np.linalg.pinv(x), y)

        return

    def predict(self, x: np.ndarray):
        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        return np.dot(x, self.beta)
