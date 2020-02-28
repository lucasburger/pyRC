
import numpy as np
from scipy import sparse


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


def expand_echo_matrix(echo=None, new_shape=None, new_matrix=None):
    if echo is None:
        return new_matrix

    if new_shape is None:
        new_shape = new_matrix.shape

    new_echo = np.zeros(shape=(echo.shape[0]+new_shape[0], echo.shape[1]+new_shape[1]), dtype=np.float)
    new_echo[:echo.shape[0], :echo.shape[1]] = echo.todense()
    if new_matrix is not None:
        new_echo[echo.shape[0]:new_echo.shape[0], echo.shape[1]:new_echo.shape[1]] = new_matrix

    if sparse.issparse(echo):
        new_echo = sparse.csr_matrix(new_echo)

    return new_echo
