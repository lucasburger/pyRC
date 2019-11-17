
from copy import copy
import numpy as np
from .output_models import GaussianElimination
from .reservoirs import BaseReservoir
from .scaler import tanh, identity
from .. import util
from .. import optimizer


class ReservoirModel(object):

    """
    This class is base model for any Reservoir computing model. Main difference will be the attribute _reservoir_class,
    which creates an EchoStateNetwork if set to ESNReservoir or a StateAffineSytem if set to SASReservoir
    """
    _reservoir_class = BaseReservoir

    def __init__(self, **kwargs):

        self.rmg = kwargs.pop('prob_dist', util.matrix_uniform)

        self.input_scaler = identity
        self.output_scaler = identity

        self.input_activation = kwargs.pop('input_activation', util.identity)
        self.input_inv_activation = kwargs.pop('input_inv_activation', util.identity)

        self.regress_input = kwargs.pop('regress_input', True)

        self.output_model = kwargs.pop(
            'output_model', GaussianElimination())

        try:
            if self.output_model.get_params()['cv'] is None:
                Warning("No crossvalidation set for output model.")
                try:
                    self.output_model.set_params(store_cv_values=True)
                except KeyError:
                    pass
        except AttributeError:
            pass

        self.W_in = kwargs.pop('W_in', None)

        self.reservoir = kwargs.get(
            'reservoir', self._reservoir_class(**kwargs))

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

        return reservoir.update(np.dot(input_array, self.W_in))

    def train(self, feature, burn_in_feature=None, burn_in_split=0.1, teacher=None,
              error_fun=util.RMSE,
              hyper_tuning=False, dimensions=None, exclude_hyper=None,
              minimizer=None):
        """
        Training of the network
        :param feature: features for the training
        :param burn_in_feature: features for the initial transient
        :param teacher: teacher signal for training
        :param error_fun: error function that should be used
        :return: tuple of minimizer result and dict with used parameter (only if hyper_tuning is True)
        """

        if burn_in_feature is None:
            if burn_in_split < 1.0:
                burn_in_ind = int(burn_in_split * feature.shape[0])
            else:
                burn_in_ind = int(burn_in_split)

            # adjust feature accordingly
            burn_in_feature = feature[:burn_in_ind, :]
            feature = feature[burn_in_ind:, :]

            # and teacher if it has been provided
            if teacher is not None:
                teacher = teacher[burn_in_ind:, :]

        # if no teacher has been provided, shift the feature by one
        if teacher is None:
            teacher = feature[1:, :]
            feature = feature[:-1, :]

        self.burn_in_feature = burn_in_feature
        self.feature = feature
        self.teacher = teacher

        # if W_in has not been provided in the __init__, set it according to the sizes
        if self.W_in is None:
            self.W_in = util.matrix_uniform(self._feature.shape[-1], self.reservoir.input_size)

        # hyper tuning if set to True
        r, result_dict = None, {}
        if hyper_tuning:
            r, result_dict = optimizer.optimizer(self, error_fun=error_fun, dimensions=dimensions,
                                                 minimizer=minimizer, exclude_hyper=exclude_hyper)

        x = self._train()
        if hyper_tuning:
            return r, result_dict
        else:
            return x, {}

    def _train(self):
        """
        This method can be used after new hyper_parameter shave been set. 
        It only uses the instance variables (burn_in-) feature and teacher
        """

        self.update(self._burn_in_feature)
        x = self.update(self._feature)
        if self.regress_input:
            x = np.hstack((self._feature, x))

        # fit output model and calculate errors
        self.output_model.fit(x, self._teacher)
        self._fitted = self.output_model.predict(x)
        self._errors = (self._fitted - self._teacher).flatten()

        return x

    def predict(self, n_predictions=0, feature=None, simulation=True, return_states=False,
                inject_error=False, num_reps=1):
        """
        let the model predict
        :param n_predictions: number of predictions in free run mode. This only works if
        number of outputs is equal to the number of inputs
        :param feature: features to predict
        :param simulation: (Boolean) if a copy of the reservoir should be used
        :param inject_error: (Boolean) if estimation error is added to the prediction
        :param num_reps: if inject_error is True, number of 
                         repetitions performed with different errors
        :return: predictions
        """

        #reservoir = self.reservoir.copy() if simulation else self.reservoir
        _feature = self.input_activation(self.input_scaler.scale(
            self.teacher[-1, :] if feature is None else feature))

        prediction = np.zeros(shape=(n_predictions, num_reps), dtype=np.float64)

        if return_states:
            states = np.zeros(shape=(n_predictions, self.size, num_reps), dtype=np.float64)

        with self.reservoir.simulate(simulation):
            for rep in range(num_reps):
                # inject error if set to true, else set errors to zeros
                if inject_error:
                    errors = np.random.choice(self._errors, n_predictions)
                else:
                    errors = np.zeros((n_predictions,))

                for i in range(n_predictions):

                    # get new regressors for output model
                    x = self.update(_feature)

                    if return_states:
                        states[i, :, rep] = x.flatten()

                    if self.regress_input:
                        x = np.hstack((_feature.flatten(), x.flatten()))

                    # predict next value
                    pred = self.output_model.predict(x.reshape(1, -1))
                    output = self.output_scaler.unscale(pred.flatten())
                    prediction[i, rep] = float(output)

                    # transform output to new input and save it in variable _feature
                    _feature = self.input_activation(self.input_scaler.scale(output)).reshape((-1, 1))
                    _feature += errors[i]

        prediction = np.mean(prediction, axis=-1)
        if return_states:
            #states = np.mean(states, axis=-1)
            pass

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

    def get_params(self):
        return {k: getattr(self.reservoir, k) for k in self.reservoir.hyper_params.keys()}

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
        return self.output_scaler.unscale(self._teacher)

    @property
    def error(self):
        try:
            cv_values = self.output_model.cv_values_
            return np.min(np.mean(cv_values, axis=0))
        except AttributeError:
            pass

        try:
            return self.output_model.error
        except AttributeError:
            return util.RMSE(self._errors)

    @property
    def fitted(self):
        return self.output_scaler.unscale(self._fitted)

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
        self._teacher = self.output_scaler.scale(x)

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
