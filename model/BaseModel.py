
from copy import copy
import numpy as np
from model.output_models import RidgeRegressionCV
from model.reservoirs import BaseReservoir
from model.scaler import tanh, identity
import util


class ReservoirModel:

    """
    This class is the self-written implementation of the Echo State Network
    """
    _reservoir_class = BaseReservoir

    def __init__(self, **kwargs):

        self.rmg = kwargs.pop('prob_dist', util.matrix_uniform)

        self.input_scaler = tanh
        self.output_scaler = identity

        self.input_activation = kwargs.pop('input_activation', np.tanh)
        self.input_inv_activation = kwargs.pop(
            'input_inv_activation', np.arctanh)

        self.output_activation = kwargs.pop('output_activation', util.identity)
        self.output_inv_activation = kwargs.pop(
            'output_inv_activation', util.identity)

        self.feed_input = kwargs.pop('feed_input', False)

        self.output_model = kwargs.pop(
            'output_model', RidgeRegressionCV(min_eigenvalue=1e-6))

        self.reservoir = kwargs.get(
            'reservoir', self._reservoir_class(**kwargs))

        self.W_in = None
        self._burned_in = False
        self._burn_in_feature = None
        self._feature = None
        self._output = None
        self._teacher = None
        self._fitted = None
        self._errors = None

    def update(self, input_array, reservoir=None):
        """
        Updates the reservoir with the given input and returns the state
        param np.ndarray input_array:
        """
        if reservoir is None:
            reservoir = self.reservoir

        if input_array.ndim == 1:
            if input_array.shape[0] == reservoir.size:
                input_array = input_array.reshape((1, -1))
            else:
                input_array = input_array.reshape((-1, 1))

        # if input_array.shape[0] == 1:
        return self.reservoir.update(np.dot(input_array, self.W_in))
        # else:
        #    return np.apply_along_axis(self.update, axis=1, arr=input_array, reservoir=reservoir)

    def train(self, feature, burn_in_feature=None, burn_in_split=0.1, teacher=None,
              error_fun=util.RMSE, hyper_tuning=False):
        """
        Training of the network
        :param feature: features for the training
        :param burn_in_feature: features for the initial transient
        :param teacher: teacher signal for training
        :param error_fun: error function that should be used
        :return: tuple of minimizer result and dict with used parameter (only if hyper_tuning is True)
        """

        if burn_in_feature is None:
            burn_in_ind = int(burn_in_split * feature.shape[0])
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
        if self.feed_input:
            x = np.hstack((self._feature, x))

        # fit output model and calculate errors
        y = self._teacher.flatten()
        self.output_model.fit(x, y)
        self._fitted = self.output_model.predict(x)
        self._errors = (self._fitted - self._teacher).flatten()

        return self

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
        _feature = self.input_activation(self.input_scaler.scale(
            self.teacher[-1] if feature is None else feature))

        return np.hstack(list(self.__prediction_generator(_feature=_feature, reservoir=reservoir, max_iter=n_predictions, inject_error=inject_error)))

    def __prediction_generator(self, _feature, reservoir, max_iter=1000, inject_error=False):

        count = 0

        while count < max_iter:

            # get new regressors for output model
            x = self.update(_feature, reservoir=reservoir)
            if self.feed_input:
                x = np.hstack((_feature, x))

            # predict next value
            output = self.output_model.predict(x.reshape(1, -1))

            # transform output to new input and save it in variable _feature
            _feature = self.output_to_input(output)
            if inject_error:
                _feature += np.random.choice(self._errors, 1)

            # increase counter
            count += 1

            yield self.output_scaler.unscale(output)

    def output_to_input(self, x):
        return self.input_activation(self.input_scaler.scale(self.output_scaler.unscale(x)))

    """
    The following are helper functions as well as attribute setters and getters
    """

    def __repr__(self):
        return f"(ReservoirModel: N={self.size}, " + repr(self.reservoir) + ")"

    @property
    def n_inputs(self):
        return self.reservoir.input_size

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

    @burn_in_feature.setter
    def burn_in_feature(self, x):
        if x.ndim == 1:
            x = x.reshape((x.shape[0], 1))
        self._burn_in_feature = self.input_activation(
            self.input_scaler.scale(x))

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
            self.W_fb = self.rmg(self.n_outputs, self.reservoir.input_size)

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