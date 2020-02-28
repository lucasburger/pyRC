

import numpy as np
import json

from .ip import *
from .multivar_polynomial import *
from .random_matrix import *
from .error_fun import *
from .mackey_glass import *
from .activation_functions import *
from .de_prado import *
from .scaler import *
from scipy.stats import norm

from . import optimizer


def update_hyperparams(old, new):
    old = {k: ([v] if not isinstance(v, list) else v) for k, v in old.items()}
    new = {k: ([v] if not isinstance(v, list) else v) for k, v in new.items()}
    d = {k: old.get(k, []) + new.get(k, []) for k in (set(list(old.keys()) + list(new.keys())))}
    return d


class ts_split:

    def __init__(self, n, test_set_size=1, test_sets=1):
        if test_set_size < 1:
            test_set_size = int(test_set_size*n)
        self.test_set_size = test_set_size
        self.n = n
        self.test_sets = test_sets

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.test_sets:
            train_start = self.i*self.test_set_size
            train_end = self.n - self.test_set_size*(self.test_sets-self.i)

            test_start = train_end
            test_end = test_start + self.test_set_size

            idtrain = np.arange(train_start, train_end, dtype=int)
            idtest = np.arange(test_start, test_end, dtype=int)
            self.i += 1
            return idtrain, idtest
        else:
            raise StopIteration


# def ts_split(n, test_set_size=1, test_sets=1):

#     if test_set_size < 1:
#         test_set_size = int(test_set_size*n)

#     i = 0
#     while i < test_sets:
#         train_start = i*test_set_size
#         train_end = n - test_set_size*(test_sets-i)

#         test_start = train_end
#         test_end = test_start + test_set_size

#         idtrain = np.arange(train_start, train_end, dtype=int)
#         idtest = np.arange(test_start, test_end, dtype=int)
#         yield idtrain, idtest
#         i += 1


def make_kwargs_one_length(func):
    def kwargs_one_length(*args, **kwargs):

        if len(kwargs) == 0:
            return func(*args, **kwargs)

        lengths = []

        for k, v in kwargs.items():
            if isinstance(v, list):
                if len(v) > 1:
                    lengths.append(len(v))

        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistend inputs to {func.__name__}: equal list lengths or length 1.")

        if len(lengths) > 0:
            ma = max(lengths)
        else:
            ma = 1

        for k, v in kwargs.items():
            if not isinstance(v, list):
                kwargs[k] = [v] * ma

        result = []
        for i in range(ma):
            new_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append(func(*args, **new_kwargs))
        return result

    return kwargs_one_length


def lyapunov_exponent(rc, error_size=1e-6, reps=100, length=17, distance=200):

    num_pred = int(1.5*distance) + length
    org_state = rc.reservoir.state.copy()
    pred = rc.predict(num_pred, simulation=True)

    LEs = np.zeros((reps,))

    for i in range(100):
        if hasattr(rc.reservoir, 'reservoirs'):
            rc.reservoir.state = org_state + np.random.uniform(low=-error_size, high=error_size, size=(rc.reservoir.size,))
        else:
            rc.reservoir._state = org_state + np.random.uniform(low=-error_size, high=error_size, size=(rc.reservoir.size,))

        pred_pert = rc.predict(num_pred, simulation=False)
        start_1 = int(0.5*distance)
        end_1 = start_1 + length

        start_2, end_2 = num_pred - length, num_pred
        d1 = np.sqrt(np.sum((pred[start_1:end_1] - pred_pert[start_1:end_1])**2))
        d2 = np.sqrt(np.sum((pred[start_2:end_2] - pred_pert[start_2:end_2])**2))
        LEs[i] = np.log(d2/d1)/distance

    return np.mean(LEs)


def univariate_likelihood(x, mu=0.0, sigma=None, sigma2=None, log=True):
    if sigma2 is None:
        sigma2 = sigma**2
    if sigma is None:
        sigma = np.sqrt(sigma2)

    x = x.flatten()
    if log:
        return np.sum(norm.logpdf(x, loc=mu, scale=sigma))
    else:
        return np.prod(norm.pdf(x, loc=mu, scale=sigma))


def save_error(filename, key, data, append=False):
    try:
        with open(filename, "r") as f:
            d = json.load(f)
    except FileNotFoundError:
        d = {}

    if key not in d.keys():
        d[key] = []

    if append:
        if not isinstance(d[key], list):
            d[key] = [d[key]]

        d[key].append(data)
    else:
        d[key] = data

    d = {k: d.get(k) for k in sorted(d)}

    with open(filename, "w") as f:
        json.dump(d, f, indent=4)

    return 1
