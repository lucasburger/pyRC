import numpy as np
from statsmodels.tsa.stattools import adfuller


def getWeights_FFD(d: float, thres: float):
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
            adf_test = adfuller(y, maxlag=1, regression='t', autolag=None)
            if adf_test[1] < alpha or d > 1.0:
                break
        except Exception:
            pass
        d += increment

    return y, d
