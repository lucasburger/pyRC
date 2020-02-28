import numpy as np


def scale(x):
    return np.log(np.sqrt(x))


def unscale(x):
    return np.square(np.exp(x))
