
from abc import abstractmethod
from numpy import tanh as tanh_org
from numpy import arctanh as arctanh_org


class BaseScaler():

    @abstractmethod
    def scale(self, x):
        pass

    @abstractmethod
    def unscale(self, x):
        pass


class tanh:
    __metaclass__ = BaseScaler

    @staticmethod
    def scale(x):
        return tanh_org(x)

    @staticmethod
    def unscale(x):
        return arctanh_org(x)


class identity:
    __metaclass__ = BaseScaler

    @staticmethod
    def scale(x):
        return x

    @staticmethod
    def unscale(x):
        return x


class InputScaler(BaseScaler):

    def __init__(self, series, min_value=-0.8, max_value=0.8):

        self.min_value = min_value
        self.max_value = max_value

        self.series_max = series.max()
        self.series_min = series.min()

        if self.series_max == self.series_min:
            raise ValueError("Constant series given to scaler")

    def scale(self, x):
        return self.min_value + (x - self.series_min) / (self.series_max - self.series_min) * (self.max_value - self.min_value)

    def unscale(self, x):
        return self.series_min + (x - self.min_value) / (self.max_value - self.min_value) * (self.series_max - self.series_min)
