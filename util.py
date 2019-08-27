import numpy as np


def matrix_normal(s1, s2=None, loc=0, scale=1):
    if isinstance(s1, tuple):
        shape = s1
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


def random_echo_matrix(size, sparsity=0.0, spectral_radius=None):
    if size == (0, 0):
        return np.zeros(shape=size, dtype=np.float64)
    w = matrix_uniform(size, size)
    w[matrix_uniform(w.shape, low=0) < sparsity] = 0
    if spectral_radius:
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        if radius > 0:
            w *= (spectral_radius / radius)
        else:
            w = np.zeros((size, size))
    return w


def ts_split(n, test_set_size=5, test_length=1):
    i = 0
    while i <= test_set_size:
        tr_s = i
        tr_e = n - test_set_size + i*test_length

        te_s = tr_e + 1
        te_e = te_s + test_length

        if te_e > n:
            break
        else:
            idtrain = np.arange(tr_s, tr_e, dtype=int)
            idtest = np.arange(te_s, te_e, dtype=int)
            yield idtrain, idtest
            i += 1


def RMSE(x, y=None):
    y = y if y is not None else np.zeros_like(x)
    return np.sqrt(np.mean(np.square(x.flatten()-y.flatten())))


def identity(x):
    return x


def MackeyGlass(l, beta=0.2, gamma=0.1, tau=17, n=10, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    x = list(np.random.uniform(0.5, 1.2, tau))
    for _ in range(l-tau):
        new_value = x[-1] + beta*x[-tau]/(1+x[-tau]**n) - gamma*x[-1]
        x.append(new_value)

    return x


def recurrence_of_reservoir(reservoir):
    layers = reservoir.layers
    connections = reservoir.connections
    ll = []
    for i in range(len(layers)):
        l = []
        for j, layer in enumerate(layers):
            if j == i:
                l.append(layer.echo)
            else:
                m = connections.get('{}-{}'.format(i, j), None)
                if m is not None:
                    l.append(m)
                else:
                    l.append(np.ndarray(shape=(layers[i].size, layer.size)))
        ll.append(l)

    x = np.block(ll)

    return x


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
        for k,v in kwargs.items():
            if isinstance(v, list):
                lengths.append(len(v))

        if len(set(lengths)) > 1:
            raise ValueError

        ma = max(lengths)
        for k,v in kwargs.items():
            if not isinstance(v, list):
                kwargs[k] = [v for _ in range(ma)]

        return func(*args, **kwargs)
    return f