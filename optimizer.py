import numpy as np
from copy import deepcopy
from datetime import datetime as dt

from skopt import dummy_minimize
from skopt.utils import use_named_args
from skopt.utils import Real

from tqdm import tqdm
import warnings


def unravel_dimensions(d):
    ret = {}
    for k, v in d.items():
        if not isinstance(v, list):
            v = [v]
        ret.update(**{str(k) + str(i): vv for i, vv in enumerate(v)})
    return ret


def rebuild_dimensions(dims):

    # result dict
    d = {}

    # create a dictionary where the key is the parameter, but without the
    # number and the value is a list of parameters with numbers
    keys = {}
    for dim in dims.keys():
        kwn = ''.join([i for i in dim if not i.isdigit()])
        keys[kwn] = keys.get(kwn, []) + [dim]

    for key, list_of_old_keys in keys.items():
        v = [round(dims[k], 4) for k in list_of_old_keys]
        d.update({key: v if len(v) > 1 else v[0]})

    return d


class OptimizationTimer(tqdm):

    def __init__(self, *args, param_names, **kwargs):
        self.param_names = param_names
        print("Optimizing hyperparameters:")
        super().__init__(*args, **kwargs)

    def __call__(self, r):
        display_dict = {'error': r.fun}
        params = dict(zip(self.param_names, r.x))
        params = rebuild_dimensions(params)
        display_dict = {**display_dict, **params}
        self.update()
        self.set_postfix(display_dict)


def optimizer(model, error_fun=None, dimensions=None, minimizer=None, exclude_hyper=None):

    hyper_dimensions = model.hyper_params

    if dimensions is not None:
        hyper_dimensions.update(dimensions)

    if exclude_hyper is not None:
        for eh in set(exclude_hyper):
            hyper_dimensions.pop(eh)

    default_minimizer = {'optimizer': dummy_minimize,
                         'n_calls': 20*len(hyper_dimensions)}

    if minimizer is not None:
        default_minimizer.update(minimizer)

    minimizer_fun = default_minimizer.pop('optimizer')

    unraveled = unravel_dimensions(hyper_dimensions)

    skopt_dimensions = [Real(name=key, low=value[0], high=value[1]) for key, value in unraveled.items()]

    @use_named_args(dimensions=skopt_dimensions)
    def _objective(**kwargs):
        e = deepcopy(model)

        hyper_params = rebuild_dimensions(kwargs)
        e.set_params(**hyper_params)
        e.reservoir.reset()

        error = e._train().error
        try:
            error = error['outofsample']
        except IndexError:
            pass

        try:
            cv_scores = e.output_model.cv_values_
            error = np.min(np.mean(cv_scores, axis=error.shape[:-1]))
        except AttributeError:
            pass

        if error is None or np.isnan(error) or np.isinf(error):
            error = 10000000

        return error

    timer = OptimizationTimer(total=default_minimizer['n_calls'],
                              param_names=list(unraveled.keys()))

    warnings.filterwarnings("ignore")
    with timer:
        r = minimizer_fun(_objective, dimensions=skopt_dimensions, callback=[timer], **default_minimizer)

    # create dictionary of optimzation results to return from the train method
    unraveled = dict(zip(unraveled.keys(), r.x))
    result_dict = rebuild_dimensions(unraveled)

    model.set_params(**result_dict)

    return r, result_dict
