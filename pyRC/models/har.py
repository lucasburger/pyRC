
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from matplotlib import pyplot as plt
from datetime import datetime as dt
try:
    from .. import util
    from .output_models_online import WRLS
except ImportError:
    from pyRC import util
    from pyRC.models.output_models_online import WRLS


def moving_average(x, length, fill_na=False):
    x = x.flatten()
    n = x.shape[0]
    strided = np.lib.stride_tricks.as_strided(x, shape=(n-length, length), strides=(x.strides[0], x.strides[0]))
    ma = np.mean(strided, axis=1, dtype=np.float64).flatten()
    if fill_na:
        ma = np.hstack([np.ones(shape=(n-ma.shape[0],))*np.nan, ma])

    return ma


class HAR:

    def __init__(self, timeframes=None):
        if timeframes is None:
            timeframes = [1, 5, 22]
        self.timeframes = timeframes
        self.max_time = max(timeframes)
        self.beta = np.zeros((4,), dtype=np.float64)
        self.series = None

    def train(self, series):
        assert np.max(series.shape) > self.max_time
        self.series = series

        X = np.vstack([moving_average(series, l, fill_na=True) for l in self.timeframes]).T

        X = add_constant(X)
        x = X[(self.max_time+1):, :]

        Y = series[self.max_time+1:].reshape((-1, 1))

        self.beta = np.linalg.pinv(x) @ Y

        return np.dot(X[-1, :], self.beta)

    def predict(self, x):
        return np.dot(x, self.beta)


class OnlineHAR:

    def __init__(self, timeframes=None, **online_learner_kwargs):
        if timeframes is None:
            timeframes = [1, 5, 22]
        self.timeframes = timeframes
        self.max_time = max(timeframes)
        self.series = None
        self.online_model = WRLS(**online_learner_kwargs)

    def train(self, series):
        series = series.flatten()
        assert series.shape[0] > self.max_time

        self.series = series

        X = np.vstack([moving_average(series, l, fill_na=True) for l in self.timeframes]).T

        X = X[(self.max_time+1):, :]

        Y = series[self.max_time+1:].reshape((-1, 1))

        return self.online_model.fit(X, Y)

    def predict(self, x):
        return np.dot(x, self.beta)

    @property
    def beta(self):
        return self.online_model.beta


if __name__ == "__main__":

    df = pd.read_csv("data/IBM_daily_RVAR.csv")
    df = df[df.close != 0.0]
    series = np.log(np.sqrt(df.close.values))
    num_pred = 1000
    n = series.shape[0]

    SCALE = 0.8
    # series = (series - np.min(series))/(np.max(series)-np.min(series))*2*SCALE - SCALE

    har = HAR(timeframes=[1, 5, 22])

    x = np.lib.stride_tricks.as_strided(series, shape=(num_pred, n-num_pred), strides=(series.strides[0], series.strides[0]))

    target = series[-num_pred:]

    pred = np.apply_along_axis(har.train, axis=1, arr=x)

    qlike = util.QLIKE(pred, target)
    mse = util.MSE(pred, target)
    expmse = util.MSE(np.exp(pred), np.exp(target))
    d = dict(qlike=qlike, mse=mse, expmse=expmse)

    util.save_error("errors.json", "har", d, append=True)

    plt.plot(target)
    plt.plot(pred)
    plt.legend(['target', 'fitted'])
    plt.title("QLIKE = {:0.6f}, MSE = {:0.6f}, expMSE = {:0.6f}".format(qlike, mse, expmse))
    plt.show()
