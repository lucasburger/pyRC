
import numpy as np
from model.output_models import RidgeRegressionCV
from model.reservoirs import LeakyReservoir, DeepReservoir
from model.scaler import tanh, identity
import util


class EchoStateNetwork:

    """
    This class is the self-written implementation of the Echo State Network
    """

    _burned_in = False

    W_in = None
    W_fb = None

    _reservoir_class = LeakyReservoir

    def __init__(self, **kwargs):

        self.random_seed = kwargs.get('random_seed', None)
        self.rmg = kwargs.get('prob_dist', util.matrix_uniform)

        self.input_scaler = tanh
        self.output_scaler = identity

        self.input_activation = kwargs.get('input_activation', np.tanh)
        self.input_inv_activation = kwargs.get('input_inv_activation', np.arctanh)

        self.output_activation = kwargs.get('output_activation', util.identity)
        self.output_inv_activation = kwargs.get('output_inv_activation', util.identity)

        self.feed_input = kwargs.get('feed_input', False)

        self.reservoir = kwargs.get('reservoir', self._reservoir_class(**kwargs))

        self.output_model = kwargs.get('output_model', RidgeRegressionCV(min_eigenvalue=1e-6))

        self.W_in = None
        self._burn_in_feature = None
        self._feature = None
        self._output = None
        self._teacher = None
        self._fitted = None
        self._errors = None

    def train(self, feature, burn_in_feature=None, burn_in_split=0.1, teacher=None):
        """
        Training of the network
        :param burn_in_feature: features for the initial transient
        :param feature: features for the training
        :param teacher: teacher signal for training
        :param error_fun: error function that should be used
        :return: tuple of minimizer result and dict with used parameter (only if hyper_tuning is True)
        """

        if burn_in_feature is None:
            burn_in_ind = burn_in_split * feature.shape[0]
            burn_in_feature = feature[:burn_in_ind, :]
            feature = feature[burn_in_ind:, :]

        if teacher is None:
            teacher = feature[1:, :]
            feature = feature[:-1, :]

        self.burn_in_feature = burn_in_feature
        self.feature = feature
        self.teacher = teacher

        self.W_in = util.matrix_uniform(self._feature.shape[-1], self.reservoir.input_size)

        self.update(self._burn_in_feature)
        x = self.update(self._feature)
        if self.feed_input: x = np.hstack((self._feature, x))
        
        # fit output model and calculate errors
        self.output_model.fit(x, self._teacher.flatten())
        self._fitted = self.output_model.predict(x)
        self._errors = (self._fitted - self._teacher).flatten()

        return self

    def update(self, input_array, reservoir=None):
        if reservoir is None: reservoir = self.reservoir
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            return self.reservoir.update(np.dot(input_array, self.W_in))
        else:
            return np.apply_along_axis(self.update, axis=1, arr=input_array, reservoir=reservoir)

    def predict(self, n_predictions=0, feature=None, simulation=True, inject_error=False):
        """
        let the model predict
        :param n_predictions: number of predictions in free run mode. This only works if
        number of outputs is equal to the number of inputs
        :param feature: features to predict
        :param simulation: (Boolean) if a copy of the reservoir should be used
        :param inject_error: (Boolean) if estimation error is added to the prediction
        :return: predictions
        """

        reservoir = self.reservoir.copy() if simulation else self.reservoir
        _new_feature = self.input_activation(self.input_scaler.scale(self.teacher[-1] if feature is None else feature))

        return np.hstack(list(self.__prediction_generator(_first_feature=_new_feature, reservoir=reservoir, max_iter=n_predictions, inject_error=inject_error)))

    def __prediction_generator(self, _first_feature=None, reservoir=None, simulation=True, max_iter=100, inject_error=False):

        _feature = self._teacher[-1] if _first_feature is None else _first_feature
        reservoir = (self.reservoir.copy() if simulation else self.reservoir) if reservoir is None else reservoir
        count = 0
        while True and count < max_iter:
            x = self.update(_feature, reservoir=reservoir)
            if self.feed_input: x = np.hstack((_feature, x))
            output = self.output_model.predict(x)
            _feature = self.output_to_input(output) + (np.random.choice(self._errors, 1) if inject_error else 0)
            count += 1
            yield self.output_scaler.unscale(output)

    def output_to_input(self, x):
        return self.input_activation(self.input_scaler.scale(self.output_scaler.unscale(x)))

    """
    The following are helper functions as well as attribute setters and getters
    """

    def get_features(self, ind):
        if self._feature is not None:
            if self._feature.ndim == 1:
                return self._feature[ind]
            else:
                return self._feature[ind, :]

    def get_teacher(self, ind):
        if self._teacher is not None:
            if self._teacher.ndim == 1:
                return self._teacher[ind]
            else:
                return self._teacher[ind, :]

    @property
    def n_inputs(self):
        return self._feature.shape[-1]

    @property
    def n_outputs(self):
        return self._teacher.shape[-1]

    @property
    def bias(self):
        return self.reservoir.bias

    @property
    def spectral_radius(self):
        return self.reservoir.spectral_radius

    @property
    def size(self):
        return self.reservoir.size

    @property
    def leak(self):
        return self.reservoir.leak

    @property
    def sparsity(self):
        return self.reservoir.sparsity

    @property
    def burn_in_feature(self):
        return self.input_scaler.unscale(self.input_inv_activation(self._burn_in_feature))

    @property
    def feature(self):
        return self.input_scaler.unscale(self.input_inv_activation(self._feature))

    @property
    def teacher(self):
        return self.output_scaler.unscale(self.output_inv_activation(self._teacher))

    @property
    def error(self):
        return util.RMSE(self._errors)

    @property
    def fitted(self):
        return self.output_scaler.unscale(self.output_inv_activation(self._fitted))

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, x):
        if x is not None:
            self._random_seed = x
            np.random.seed(x)

    @burn_in_feature.setter
    def burn_in_feature(self, x):
        if x.ndim == 1:
            x = x.reshape((x.shape[0], 1))
        self._burn_in_feature = self.input_activation(self.input_scaler.scale(x))

    @feature.setter
    def feature(self, x):
        if x.ndim == 1:
            x = x.reshape((x.shape[0], 1))
        self._feature = self.input_activation(self.input_scaler.scale(x))

    @teacher.setter
    def teacher(self, x):
        if x.ndim == 1:
            x = x.reshape((x.shape[0], 1))
        self._teacher = self.output_scaler.scale(x)
        if not self._burned_in:
            self.W_fb = self.rmg(self.n_outputs, self.reservoir.size)

    @sparsity.setter
    def sparsity(self, x):
        self.reservoir.sparsity = x

    @bias.setter
    def bias(self, x):
        self.reservoir.bias = x

    @leak.setter
    def leak(self, x):
        self.reservoir.leak = x

    @spectral_radius.setter
    def spectral_radius(self, x):
        self.reservoir.spectral_radius = x

    @fitted.setter
    def fitted(self, x):
        self._fitted = x


class deepESN(EchoStateNetwork):
    _reservoir_class = DeepReservoir

    def add_layer(self, *args, **kwargs):
        self.reservoir.add_layer(*args, **kwargs)

    @property
    def layers(self):
        return getattr(self.reservoir, 'layers', None)