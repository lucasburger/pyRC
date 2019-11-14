import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import sparse
import numba
from math import factorial as fac
from itertools import product


def update_hyperparams(old, new):
    old = {k: ([v] if not isinstance(v, list) else v) for k, v in old.items()}
    new = {k: ([v] if not isinstance(v, list) else v) for k, v in new.items()}
    d = {k: old.get(k, []) + new.get(k, []) for k in (set(list(old.keys()) + list(new.keys())))}
    return d


def matrix_normal(s1, s2=None, loc=0, scale=1):

    if isinstance(s1, tuple):
        shape = s1
    elif s2 is None:
        shape = (s1,)
    else:
        shape = (s1, s2)

    return np.random.normal(loc=loc, scale=scale, size=shape)


def matrix_uniform(s1, s2=None, low=-1, high=1):
    if isinstance(s1, tuple):
        shape = s1
    elif s2 is None:
        shape = (s1,)
    else:
        shape = (s1, s2)

    return np.random.uniform(low=low, high=high, size=shape)


def random_echo_matrix(size, sparsity=0.0, spectral_radius=None, prob_dist='uni'):

    if prob_dist == 'uni':
        matrix_fun = matrix_uniform
    elif prob_dist in ['norm', 'normal']:
        matrix_fun = matrix_normal

    if not isinstance(size, tuple):
        size = (size, size)

    if size == (0, 0):
        return np.zeros(shape=size, dtype=np.float64)

    w = matrix_fun(size)
    w[matrix_uniform(w.shape, low=0) < sparsity] = 0
    if spectral_radius:
        if size[0] != size[1]:
            radius = np.max(np.abs(np.linalg.svd(w, compute_uv=False)))
        else:
            radius = np.max(np.abs(np.linalg.eigvals(w)))

        if radius > 0:
            w *= (spectral_radius / radius)
        else:
            w = np.zeros(size)

    return w


def ts_split(n, test_set_size=1, test_sets=1):

    if test_set_size < 1:
        test_set_size = int(test_set_size*n)

    i = 0
    while i < test_sets:
        train_start = i*test_set_size
        train_end = n - test_set_size*(test_sets-i)

        test_start = train_end
        test_end = test_start + test_set_size

        idtrain = np.arange(train_start, train_end, dtype=int)
        idtest = np.arange(test_start, test_end, dtype=int)
        yield idtrain, idtest
        i += 1


def MSE(x, y):
    return np.mean((x.flatten() - y.flatten())**2)


def RMSE(x, y=None):
    if y is None:
        y = np.zeros_like(x)
    return np.sqrt(MSE(x.flatten(), y.flatten()))


def NRMSE(x, y):
    return RMSE(x, y)/np.var(y)


def identity(x):
    return x


def MackeyGlass(l, beta=0.2, gamma=0.1, tau=17, n=10, drop_out=0.1, initial_condition=None, random_seed=None):
    if random_seed or initial_condition is None:
        np.random.seed(random_seed)
        initial_condition = np.random.uniform(0.5, 1.2, tau)
    elif not isinstance(initial_condition, np.ndarray):
        initial_condition = np.asarray(initial_condition).flatten()

    if initial_condition.shape[0] != tau:
        initial_condition = np.hstack((initial_condition, np.zeros((tau-initial_condition.shape[0],))))

    assert len(initial_condition) >= tau
    assert l > tau

    l = int(l/(1-drop_out))

    mg = MackeyGlassNumba(l, initial_condition, beta=beta, gamma=gamma, tau=tau, n=n)
    return mg[int(l*drop_out):]


@numba.njit(parallel=True)
def MackeyGlassNumba(l: int, initial_condition: np.ndarray, beta: float = 0.2, gamma: float = 0.1, tau: int = 17, n: int = 10):

    x = np.zeros(shape=(l,))
    x[:tau] = initial_condition[-tau:]

    for i in range(tau, l):
        x[i] = x[i-1] + beta*x[i-tau]/(1+x[i-tau]**n) - gamma*x[i-1]

    return x


# this just compiles MackeyGlassNumba to be used later on
#x = MackeyGlass(100)


def expand_echo_matrix(echo=None, new_shape=None, new_matrix=None):
    if echo is None:
        return new_matrix

    if new_shape is None:
        new_shape = new_matrix.shape

    new_echo = np.zeros(shape=(echo.shape[0]+new_shape[0], echo.shape[1]+new_shape[1]), dtype=np.float)
    new_echo[:echo.shape[0], :echo.shape[1]] = echo
    if new_matrix is not None:
        new_echo[echo.shape[0]:new_echo.shape[0], echo.shape[1]:new_echo.shape[1]] = new_matrix
    return new_echo


def self_activation(radius=1):
    def ret_fun(x):
        return radius * x / np.linalg.norm(x)
    return ret_fun


def make_kwargs_one_length(func):
    def f(*args, **kwargs):
        lengths = []
        if len(kwargs) == 0:
            return func(*args, **kwargs)

        for k, v in kwargs.items():
            if isinstance(v, list):
                lengths.append(len(v))
            else:
                kwargs[k] = [v]
                lengths.append(1)

        if len(set(lengths)) > 1:
            raise ValueError

        ma = max(lengths)
        for k, v in kwargs.items():
            if not isinstance(v, list):
                kwargs[k] = [v for _ in range(ma)]

        return func(*args, **kwargs)
    return f


def getWeights_FFD(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    return w


def fracDiff(x, d=0.0, thres=1e-5):
    """
    param x: One dimensional array-like
    param d >= 0.0 (default): Fractional differentiation parameter
    param thres: threshold for weights cut of, default = 1e-5
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]
    """

    x = np.array(x, dtype=np.float64).ravel()

    w = getWeights_FFD(d, thres)
    width = len(w)

    a = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0]-width, width), strides=(x.strides[0],)*2)

    return np.dot(a, w)


def minLossFracDiff(series, increment=0.01, alpha=0.01):
    d = 0.0
    while True:
        try:
            y = fracDiff(series, d)
            adf_test = adfuller(y, maxlag=1, regression='c', autolag=None)
            if adf_test[1] < alpha or d > 1.0:
                break
        except Exception:
            pass
        d += increment

    return y, d


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
    def random(cls, shape, input_dim=1, order=1, sparsity=None, spectral_radius=None):

        if not isinstance(shape, tuple):
            shape = (shape,)*2

        if sparsity is None:
            sparsity = 1-float(10/np.min(shape))

        r = cls(input_dim, order)

        num_params = r.num_params

        if spectral_radius:
            spectral_radius /= num_params

        coeff = [random_echo_matrix(size=shape, sparsity=sparsity, spectral_radius=spectral_radius)
                 for _ in range(num_params)]
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


class MultivariateTrigoPolynomial(MultivariateFunction):

    def param_factors(self, base):
        r"""
        Combinations of factors with sum of exponents less than the order of the polynomial
        e.g.: sum_{i_k \in {0,...,order} sum(i_k) < order} i_k * x_k
        """
        factors = list(product(range(self.order+1), repeat=base.shape[0]))
        factors = list(filter(lambda x: sum(x) <= self.order, factors))
        x = np.hstack([np.cos(np.sum(base * p)) for p in factors])
        return x
