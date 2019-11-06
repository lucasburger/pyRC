
from copy import copy
import numpy as np
from model.output_models import MinEigenvalueRegression
from model.reservoirs import BaseReservoir
from model.scaler import tanh, identity
import util
import optimizer


class ReservoirModel(object):

    """
    This class is base model for any Reservoir computing model. Main difference will be the attribute _reservoir_class,
    which creates an EchoStateNetwork if set to ESNReservoir or a StateAffineSytem if set to SARReservoir
    """
    _reservoir_class = BaseReservoir

    def __init__(self, **kwargs):

        self.rmg = kwargs.pop('prob_dist', util.matrix_uniform)

        self.input_scaler = identity
        self.output_scaler = identity

        self.input_activation = kwargs.pop('input_activation', util.identity)
        self.input_inv_activation = kwargs.pop('input_inv_activation', util.identity)

        self.output_activation = kwargs.pop('output_activation', util.identity)
        self.output_inv_activation = kwargs.pop('output_inv_activation', util.identity)

        self.feed_input = kwargs.pop('feed_input', True)

        self.output_model = kwargs.pop(
            'output_model', MinEigenvalueRegression(min_eigenvalue=1e-6))

        # try:
        #     if self.output_model.get_params()['cv'] is None:
        #         Warning("No crossvalidation set for output model.")
        # except AttributeError:
        #     pass

        # try:
        #     self.output_model.set_params(store_cv_values=True)
        # except AttributeError:
        #     pass

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
              error_fun=util.RMSE, hyper_tuning=False,
              dimensions=None, minimizer=None, exclude_hyper=None):
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
            if teacher is not None:
                teacher = teacher[burn_in_ind:, :]

        if teacher is None:
            teacher = feature[1:, :]
            feature = feature[:-1, :]

        self.burn_in_feature = burn_in_feature
        self.feature = feature
        self.teacher = teacher

        self.W_in = util.matrix_uniform(self._feature.shape[-1], self.reservoir.input_size)

        r, result_dict = None, {}
        if hyper_tuning:
            r, result_dict = optimizer.optimizer(self, error_fun=error_fun, dimensions=dimensions,
                                                 minimizer=minimizer, exclude_hyper=exclude_hyper)

        self._train()

        return r, result_dict

    def _train(self):
        """
        This method can be used after new hyper_parameter shave been set. 
        It only uses the instance variables (burn_in-) feature and teacher
        """

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

    def predict(self, n_predictions=0, feature=None, simulation=True, return_states=False, inject_error=False):
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
            self.teacher[-1, :] if feature is None else feature))

        prediction = np.zeros(shape=(n_predictions,), dtype=np.float64)

        if return_states:
            states = np.zeros(shape=(n_predictions, reservoir.size), dtype=np.float64)
        else:
            states = None

        for i in range(n_predictions):

            # get new regressors for output model
            x = self.update(_feature, reservoir=reservoir)
            if return_states:
                states[i, :] = x
            if self.feed_input:
                x = np.hstack((_feature.flatten(), x))

            # predict next value
            pred = self.output_model.predict(x.reshape(1, -1))
            output = self.output_scaler.unscale(pred.flatten())
            prediction[i] = float(output)

            # transform output to new input and save it in variable _feature
            _feature = self.input_activation(self.input_scaler.scale(output)).reshape((-1, 1))

            # inject error if set to true
            if inject_error:
                _feature += np.random.choice(self._errors, 1)

        if return_states:
            return prediction, states
        else:
            return prediction

    def output_to_input(self, x):
        return self.input_activation(self.input_scaler.scale(self.output_scaler.unscale(x)))

    def set_params(self, **params):
        for name, value in params.items():
            if hasattr(self.reservoir, name):
                setattr(self.reservoir, name, value)
            elif hasattr(self, name):
                self.__setattr__(name, value)

    """
    The following are helper functions as well as attribute setters and getters
    """

    def __repr__(self):
        return "({}: N={}, ".format(self.__class__.__name__, self.size) + repr(self.reservoir) + ")"

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
    def hyper_params(self):
        return self.reservoir.hyper_params

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
        try:
            return self.output_model.error
        except AttributeError:
            return util.RMSE(self._teacher, self._fitted)
        # return util.RMSE(self._errors)

    @property
    def fitted(self):
        return self.output_scaler.unscale(self.output_inv_activation(self._fitted))

    @burn_in_feature.setter
    def burn_in_feature(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        self._burn_in_feature = self.input_activation(self.input_scaler.scale(x))

    @feature.setter
    def feature(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        self._feature = self.input_activation(self.input_scaler.scale(x))

    @teacher.setter
    def teacher(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        self._teacher = self.output_activation(self.output_scaler.scale(x))
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
