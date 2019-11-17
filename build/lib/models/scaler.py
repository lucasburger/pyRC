
from abc import abstractmethod
from numpy import tanh as tanh_org
from numpy import arctanh as arctanh_org

import inspect


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
