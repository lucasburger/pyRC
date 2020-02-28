import numpy as np


def MSE(x, y):
    return np.mean((x.flatten() - y.flatten())**2)


def RMSE(x, y=None):
    if y is None:
        y = np.zeros_like(x)
    return np.sqrt(MSE(x.flatten(), y.flatten()))


def NRMSE(x, y=None):
    if y is None:
        v = np.var(x)
    else:
        v = np.var(y)
    return RMSE(x, y)/v


def QLIKE(true_value, forecast, log=False, square=False):

    if log:
        true_value, forecast = np.exp(true_value), np.exp(forecast)
    if square:
        true_value, forecast = np.square(true_value), np.square(forecast)

    assert np.all(true_value > 0)
    assert np.all(forecast > 0)

    return np.mean(true_value/forecast - np.log(true_value/forecast) - 1)
