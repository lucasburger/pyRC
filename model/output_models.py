
import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifier, Lars
import sklearn.linear_model
from copy import deepcopy
import util
from abc import ABCMeta, abstractmethod


class BaseOutputModel:
    __metaclass__ = ABCMeta
    error = None

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

    @property
    def online(self):
        return hasattr(self, 'update')


class RidgeRegressionCV(BaseOutputModel):

    _alphas = []

    def __init__(self, intercept=True, min_eigenvalue=None, cv_split=None, alphas=None, **kwargs):

        self.intercept = intercept
        self.min_eigenvalue = min_eigenvalue

        if alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = [-3, -1, 2, 5]

        self.cv_split = cv_split

        self.size = None
        self.beta = None
        self.regularize_lambda = None
        self.fitted = None
        self.error = None

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
            y = y.reshape((y.shape[0], 1))

        xx = x.T @ x
        if any(np.isnan(xx.flatten())) or any(np.isinf(xx.flatten())):
            if y.ndim == 2:
                self.beta = np.zeros(shape=(self.size, y.shape[1]))
            else:
                self.beta = np.zeros(shape=(self.size,))
            self.regularize_lambda = 0
            return

        if self.min_eigenvalue is not None:
            self.regularize_lambda = self.min_eigenvalue - min(np.linalg.eigvals(xx))

            if isinstance(self.regularize_lambda, complex):
                self.regularize_lambda = np.real(self.regularize_lambda)

            self.beta = np.linalg.pinv(xx + self.regularize_lambda * self.size * np.eye(self.size)) @ x.T @ y
            self.fitted = np.dot(x, self.beta)
            self.error = util.RMSE(self.fitted, y)

        else:
            error = {}

            if self.cv_split is None:
                cv_split = util.ts_split(x.shape[0])
            else:
                cv_split = self.cv_split

            for a in self.alphas:
                error_alpha = []
                for split in cv_split:
                    x_train, y_train = x[split[0], :], y[split[0], :]
                    x_test, y_test = x[split[1], :], y[split[1], :]

                    xxt = x_train.T @ x_train
                    if any(np.isnan(xxt.flatten())) or any(np.isinf(xxt.flatten())):
                        if y.ndim == 2:
                            self.beta = np.zeros(shape=(self.size, y.shape[1]))
                        else:
                            self.beta = np.zeros(shape=(self.size,))
                        error_alpha.append(np.inf)
                        continue

                    size = x_train.shape[1]
                    try:
                        beta = np.linalg.pinv(xxt + a * np.eye(size)) @ x_train.T @ y_train
                    except Exception:
                        beta = np.zeros((size, ))

                    y_fit = np.dot(x_test, beta)
                    error_alpha.append(error_fun(y_fit, y_test))

                error[a] = np.mean(error_alpha)

            self.regularize_lambda = min(error, key=error.get)

            if isinstance(self.regularize_lambda, complex):
                self.regularize_lambda = np.real(self.regularize_lambda)

            self.beta = np.linalg.pinv(xx + self.regularize_lambda * np.eye(self.size)) @ x.T @ y
            self.fitted = np.dot(x, self.beta)
            self.error = min(error.values())

        return

    def predict(self, x):
        x = self.add_intercept(x)
        self.check_input(x)
        return np.dot(x, self.beta)

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, x):
        if isinstance(x, np.ndarray):
            self._alphas = x

        if not isinstance(x, list):
            self._alphas = [x]
        else:
            self._alphas = x

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, other):
        self._size = other


# class TimeSeriesCV(RidgeCV):

#     def __init__(self, *args, **kwargs):
#         self.error = None
#         self.fitted = None
#         super().__init__(*args, **kwargs)

#     def fit(self, x, y, **kwargs):
#         # train the super model
#         r = super().fit(x, y, **kwargs)

#         # predict and store error
#         self.fitted = super().predict(x)

#         self.error = util.RMSE(self.fitted, y)

#         return r

#     def predict(self, x):
#         if x.ndim == 1:
#             if x.shape[0] == self.coef_.shape[-1]:
#                 x = x.reshape((1, x.shape[0]))
#             else:
#                 x = x.reshape((x.shape[0], 1))

#         return super().predict(x)


# class LogRegCV(RidgeClassifier):

#     def __init__(self, *args, **kwargs):
#         self.error = None
#         self.fitted = None
#         super().__init__(*args, **kwargs)

#     def fit(self, x, y, **kwargs):
#         # train the super model
#         r = super().fit(x, np.ravel(y), **kwargs)

#         # predict and store error
#         self.fitted = super().predict(x)

#         self.error = np.mean(self.fitted != y)

#         return r

#     def predict(self, x):
#         if x.ndim == 1:
#             if x.shape[0] == self.coef_.shape[-1]:
#                 x = x.reshape((1, x.shape[0]))
#             else:
#                 x = x.reshape((x.shape[0], 1))

#         return super().predict(x)


# class ElasticNetCV(sklearn.linear_model.ElasticNetCV):

#     def __init__(self, *args, ts_split=None, cv_alphas=None, **kwargs):
#         self.cv_alphas = cv_alphas if cv_alphas is None else np.arange(0.1, 1.1, 0.1)
#         self.ts_split = ts_split
#         super().__init__(*args, **kwargs)

#     def fit(self, x, y, error_fun=util.RMSE, **kwargs):

#         # cv_split = self.ts_split if self.ts_split is not None else util.ts_split(x.shape[0])

#         # error = dict()

#         # for a in self.cv_alphas:
#         #     error_alpha = []
#         #     for split in cv_split:
#         #         x_train, y_train = x[split[0], :], y[split[0], :]
#         #         x_test, y_test = x[split[1], :], y[split[1], :]

#         #         y_fit = deepcopy(self).fit(x_train, y_train).predict(x_test)
                    
#         #         error_alpha.append(error_fun(y_fit, y_test))

#         #         error[a] = np.mean(error_alpha)

#         #     self.alpha = min(error, key=error.get)

#         super().fit(x, y)
#         self.fitted = super().predict(x)
#         self.error = error_fun(self.fitted, y)
#         return self

