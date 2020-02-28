
import numpy as np


def identity(x):
    return x


def self_activation(radius=1):
    def ret_fun(x):
        return radius * x / np.linalg.norm(x)
    return ret_fun


def log_activation(x):
    return np.sign(x)*np.log(1+np.abs(x))
