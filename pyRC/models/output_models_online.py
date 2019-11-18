
import numpy as np
from .output_models import BaseOutputModel


class WeightedRegularizedRecursiveLeastSquares(BaseOutputModel):

    def __init__(self, size=1, lambda_w=1.0, regularization=0.0, add_intercept=True):
        self.lambda_w = lambda_w
        self.regularization = regularization
        self.add_intercept = add_intercept
        self.beta = np.zeros((size, ), dtype=np.float64)
        self._gamma = np.eye(size, dtype=np.float64)
        self._fitted = np.zeros((1,))
        self._errors = np.zeros((1,))

    def _update_size(self, size):
        self._gamma = 1/(1+self.regularization) * np.eye(size)
        self.beta = np.zeros((size, ))

    def fit(self, x, y):
        if x.ndim == 1:
            x = x.reshape((1, -1))

        if y.ndim == 1:
            y = y.reshape((1, -1))

        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))

        self._update_size(x.shape[1])

        self._fitted = np.zeros_like(y)

        for i in range(x.shape[0]):
            self._fitted[i] = self.predict(x[i, :])
            self.update_weight(x[i, :], y[i, :])
        self._errors = self._fitted - y
        return self._fitted

    def update_weight(self, x, y):
        if float(x[0]) != 1.0 and self.add_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        self._gamma -= (self._gamma @ np.outer(x, x) @ self._gamma) / (self.lambda_w + x.T @ self._gamma @ x)
        self.beta -= (self._gamma @ x * (np.dot(x, self.beta) - y)).flatten()
        return

    def predict(self, x):
        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        return np.dot(x, self.beta)

    @property
    def error(self):
        return np.mean(self._errors)


WRRLS = WeightedRegularizedRecursiveLeastSquares


class RecursiveLeastSquares(WRRLS):

    def __init__(self, size=1, add_intercept=True):
        super().__init__(size=size, add_intercept=add_intercept)


RLS = RecursiveLeastSquares


class RegularizedRecursiveLeastSquares(WRRLS):

    def __init__(self, size=1, add_intercept=True, regularization=0.0):
        super().__init__(size=size, add_intercept=True, regularization=regularization)


RRLS = RegularizedRecursiveLeastSquares


class WeightedRecursiveLeastSquares(WRRLS):

    def __init__(self, size=1, add_intercept=True, lambda_w=1.0):
        super().__init__(size=size, add_intercept=add_intercept, lambda_w=lambda_w)


WRLS = WeightedRecursiveLeastSquares
