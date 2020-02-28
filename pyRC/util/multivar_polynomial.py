import numpy as np
from .random_matrix import random_echo_matrix
from itertools import product


class MultivariateFunction:

    def __init__(self, input_dim, order, coeff=None):
        self.coeff = coeff
        self.input_dim = input_dim
        self.order = order
        self.size = self.num_params

    def __call__(self, x):
        """
        Polynomials will be evaluated as follows:
        Assume p of order 3: R^2 ---> R^n, x=(x1, x2) |--> p(x)
        p(x) =  a_{0,0} + a_{0,1}*x2 + a_{0,2}*x2^2 +
                a_{1,0}*x1 + a_{1,1}*x1*x2 + a_{1,2}*x1*x2^2
                a_{2,0}*x1^2 + a_{2,1}*x1^2*x2
        with a_{i,j} in R^n for all i,j <= 2
        """
        if x.ndim == 3:
            raise ValueError

        if x.ndim == 1:
            if x.shape[0] == self.input_dim:
                return self._eval(x)
            elif self.input_dim == 1:
                x = x.reshape((-1, 1))

        return np.apply_along_axis(self._eval, axis=x.shape.index(self.input_dim), arr=x)

    def _eval(self, x):
        return np.einsum('ij..., i->j...', self.coeff, self.param_factors(x))

    def __repr__(self):
        return "{}: order={}, input_dim={}".format(self.__class__.__name__, self.order, self.input_dim)

    @classmethod
    def random(cls, shape, input_dim=1, order=1, sparsity=None, spectral_radius=None, matrix_type='full'):

        if not isinstance(shape, tuple):
            shape = (shape,)*2

        if sparsity is None:
            sparsity = 0.0

        r = cls(input_dim, order)

        num_params = r.num_params

        if spectral_radius:
            spectral_radius /= num_params

        if matrix_type == 'full':
            coeff = [random_echo_matrix(size=shape, sparsity=sparsity, spectral_radius=spectral_radius)
                     for _ in range(num_params)]
        elif matrix_type == 'diag':
            coeff = []
            for _ in range(num_params):
                m = np.diag(np.random.random((shape[0],)))
                m /= spectral_radius/np.max(np.abs(np.linalg.eigvals(m)))
                coeff.append(m)
        r.coeff = np.array(coeff)
        return r

    @property
    def num_params(self):
        return len(self.param_factors(np.zeros(shape=(self.input_dim, ))))

    def param_factors(self, base):
        raise NotImplementedError

    @property
    def spectral_radius(self):
        self._spectral_radius

    @spectral_radius.setter
    def spectral_radius(self, other):
        if isinstance(other, list) and len(other) == self.size:
            for i, sr in enumerate(other):
                self.coeff[i, ...] *= sr
        elif other > 0:
            self.coeff *= other**(1/self.size)
        self._spectral_radius = other


class MultivariatePolynomial(MultivariateFunction):

    def param_factors(self, base):
        r"""
        Combinations of powers with sum of exponents less than the order of the polynomial
        e.g.: prod_{i_k \in {0,...,order} sum(i_k) < order} x_{k}^{i_k}
        """
        powers = list(product(range(self.order+1), repeat=base.shape[0]))
        powers = list(filter(lambda x: sum(x) <= self.order, powers))
        x = np.hstack([np.product(np.power(base, p)) for p in powers])
        return x


class MultivariateTrigoPolynomial(MultivariatePolynomial):

    def param_factors(self, base):
        r"""
        Combinations of factors with sum of exponents less than the order of the polynomial
        e.g.: np.cos(sum_{i_k \in {0,...,order} sum(i_k) < order} i_k * x_k)
        """

        factors = list(product(range(self.order+1), repeat=base.shape[0]))
        factors = list(filter(lambda x: sum(x) <= self.order, factors))
        x = np.hstack([np.cos(np.sum(base * p)) for p in factors])

        return x
