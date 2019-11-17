
import numpy as np
from .output_models import BaseOutputModel


class RecursiveLeastSquares(BaseOutputModel):

    def __init__(self, size=1, add_intercept=True):
        self.add_intercept = add_intercept
        self.beta = np.zeros((size, 1), dtype=np.float64)
        self._gamma = np.eye(size, dtype=np.float64)
        self._errors = np.zeros((1,))

    def _update_size(self, size):
        self.beta = np.zeros((size, 1), dtype=np.float64)
        self._gamma = np.eye(size, dtype=np.float64)

    def fit(self, x, y):
        if x.ndim == 1:
            x = x.reshape((-1, 1))

        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self._update_size(x.shape[1])

        self._errors = np.zeros_like(y)

        for i in range(x.shape[0]):
            self._errors[i] = self.update_weight(x[i, :].reshape((-1, 1)), y[i, :].reshape((-1, 1)))

    def update_weight(self, x, y):
        self._gamma -= (self._gamma @ x @ x.T @ self._gamma) / (1 + x.T @ self._gamma @ x)
        self.beta -= self._gamma @ x * (x.T @ self.beta - y)
        return np.dot(x.T, self.beta)

    def predict(self, x):
        if self.add_intercept:
            x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        return np.dot(x, self.beta)

    @property
    def error(self):
        return np.mean(self._errors)


RLS = RecursiveLeastSquares


class RegularizedRecursiveLeastSquares(RecursiveLeastSquares):

    def __init__(self, *args, regularization=0.0, **kwargs):
        super().__init__(*args, ** kwargs)
        self.regularization = regularization

    def _update_size(self, size):
        self._gamma = 1/(self.regularization) * np.eye(size)
        self.beta = np.zeros((size, 1))


RRLS = RegularizedRecursiveLeastSquares


class WeightedRLS(RecursiveLeastSquares):

    def __init__(self, *args, lambda_w=1.0, **kwargs):
        self.lambda_w = lambda_w
        super().__init__(*args, **kwargs)

    def update_weight(self, x, y):
        self._gamma -= (self._gamma @ x @ x.T @ self._gamma) / (self.lambda_w + x.T @ self._gamma @ x)
        self.beta -= self._gamma @ x * (x.T @ self.beta - y)
        return np.dot(x.T, self.beta)


WRLS = WeightedRLS


class WeightedRegularizedRecursiveLeastSquares(WeightedRLS, RegularizedRecursiveLeastSquares):
    pass


WRRLS = WeightedRegularizedRecursiveLeastSquares
