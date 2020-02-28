import numpy as np
from scipy.stats import norm

IP_dict = {}


def IntrinsicPlasticity(criterion, activation, **kwargs):
    return IP_dict[criterion, activation](**kwargs)


class IntrinsicPlasticityKLTANH:

    _function = np.tanh

    def __init__(self, learning_rate=0.01, mu=0, sigma=1, update=True):
        self.update = update
        self.learning_rate = learning_rate
        self.mu = mu
        self.sigma2 = sigma**2
        self.scaling = norm.ppf(0.99999999, scale=sigma)
        self.scaling = max(self.scaling, 1.0)
        # self.scaling = 1.0
        self.a = 1
        self.b = 0

    def _fun(self, x):
        return self.scaling * self._function((self.a*x + self.b)/self.scaling)

    def update_parameters(self, x):
        y = self._fun(x)
        delta_b = -self.learning_rate * (2*y/self.scaling**2 + 1/self.sigma2 * (1 - y**2/self.scaling**2)*(y-self.mu))
        delta_a = self.learning_rate/self.a + delta_b*x
        self.b += delta_b
        self.a += delta_a

    def __call__(self, x):
        if self.update:
            self.update_parameters(x)
        return self._fun(x)


IP_dict['KL', 'tanh'] = IntrinsicPlasticityKLTANH


class IntrinsicPlasticityKLTANHshifted(IntrinsicPlasticityKLTANH):

    _function = np.tanh

    def _fun(self, x):
        return self.mu + super()._fun(x-self.mu)


class IntrinsicPlasticityKLARCSINH(IntrinsicPlasticityKLTANH):

    _function = np.arcsinh

    def _fun(self, x):
        return self._function(self.a*x + self.b)

    def _d_function(self, x):
        return 1 / np.sqrt(x**2 + 1)

    def _dd_function(self, x):
        return -x * (x**2 + 1)**(-3/2)

    def update_parameters(self, x):
        y = self._fun(x)
        d_axb = self._d_function(self.a*x + self.b)
        dd_axb = self._dd_function(self.a*x + self.b)
        delta_b = -self.learning_rate * (- dd_axb / d_axb + d_axb/self.sigma2 * (2*y - self.mu))
        delta_a = self.learning_rate/self.a + delta_b*x
        self.b += delta_b
        self.a += delta_a


IP_dict['KL', 'arcsinh'] = IntrinsicPlasticityKLARCSINH


class IntrinsicPlasticityWSTANH:

    _function = np.tanh

    def __init__(self, learning_rate=0.01, mu=0, sigma=1, update=True):
        self.update = update
        self.learning_rate = learning_rate
        self.mu = mu
        self.sigma = sigma
        self.scaling = norm.ppf(0.995, scale=sigma)
        self.a = 1
        self.b = 0

    def _fun(self, x):
        return self.scaling * self._function((self.a*x + self.b)/self.scaling)

    def update_parameters(self, x):
        y = self._fun(x)
        # delta_b = -(self.learning_rate/self.sigma2)*(-self.mu + y*(2*self.sigma2 + 1 - y**2 + self.mu*y))
        # delta_b = -self.learning_rate * (-2*y + 1/self.sigma2 * (self.scaling**2/2 * (1 - y**2) - self.mu*(1-y/self.scaling)))
        # delta_b = -self.learning_rate * (-2*y + 1/self.sigma2 * (1 - y**2/self.scaling)*(self.mu-y))
        delta_b = self.learning_rate * (1 - y**2/self.scaling**2)*(1+self.sigma * y)
        delta_a = delta_b*x
        self.b += delta_b
        self.a += delta_a

    def __call__(self, x):
        if self.update:
            self.update_parameters(x)
        return self._fun(x)


IP_dict['WS', 'tanh'] = IntrinsicPlasticityWSTANH
