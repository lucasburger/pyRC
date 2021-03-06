
from copy import copy
import numpy as np
from .output_models import GaussianElimination
from .reservoirs import BaseReservoir
from ..util.scaler import tanh, identity
from .. import util
from ..util import optimizer


class ReservoirModel(object):

    """
    This class is base model for any Reservoir computing model. Main difference will be the attribute _reservoir_class,
    which creates an EchoStateNetwork if set to ESNReservoir or a StateAffineSytem if set to SASReservoir
    """
    _reservoir_class = BaseReservoir

    def __init__(self, **kwargs):

        self.rmg = kwargs.pop('prob_dist', util.matrix_uniform)

        self.input_scaler = kwargs.pop('input_scaling', util.identity)
        self.output_scaler = kwargs.pop('output_scaling', util.identity)

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

        if self.W_in is None:
            self.W_in = util.matrix_uniform(input_array.shape[-1], reservoir.input_size)

        if input_array.ndim == 1:
            if input_array.shape[0] == reservoir.size:
                input_array = input_array.reshape((1, -1))
            else:
                input_array = input_array.reshape((-1, 1))

        return reservoir.update(np.dot(input_array, self.W_in))

    def train(self, feature, burn_in_feature=None, burn_in_split=0.1, teacher=None,
              error_fun=util.RMSE,
              hyper_tuning=False, dimensions=None, exclude_hyper=None,
              minimizer=None, bigger_size=None):
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

            if bigger_size is not None:
                self.reservoir.size = bigger_size
                self.W_in = util.matrix_uniform(self._feature.shape[-1], self.reservoir.input_size)

        x = self._train()

        if hyper_tuning:
            return r, result_dict
        else:
            return x

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
        fitted = self.output_model.predict(x)
        if isinstance(fitted, tuple):
            self._fitted = fitted[0]
        else:
            self._fitted = fitted

        self._errors = (self._fitted - self._teacher).flatten()

        return x

    def predict(self, n_predictions=1, feature=None, simulation=True, return_states=False,
                inject_error=False, num_reps=1,
                return_extra=False):
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

        if feature is not None and feature.ndim == 2:
            def _help_predict(x):
                return self.predict(feature=x, simulation=False)
            return np.apply_along_axis(_help_predict, axis=1, arr=feature)

        if num_reps > 1:
            inject_error = True

        # reservoir = self.reservoir.copy() if simulation else self.reservoir
        _feature = self.input_activation(self.input_scaler.scale(
            self.teacher[-1, :] if feature is None else feature)
        )

        # allocate storage for predictions
        prediction = np.zeros((n_predictions, num_reps), dtype=np.float64)

        # allocate storage for states if return_states=True or if num_reps > 1 and no simulation,
        # as then, we need to save the states to set the state of the reservoir after the repetitions
        if return_states or num_reps > 1 and not simulation:
            states = np.zeros(shape=(n_predictions, self.size, num_reps), dtype=np.float64)
        else:
            states = None

        # loop over prediction paths
        for rep in range(num_reps):
            with self.reservoir.simulate(simulation):

                # inject error if set to true, else set errors to zeros
                if inject_error:
                    errors = np.random.choice(self._errors, n_predictions)
                else:
                    errors = np.zeros((n_predictions,))

                # predict n_predictions times
                for i in range(n_predictions):

                    # get new regressors for output model
                    x = self.update(_feature)

                    # save states if needed
                    if states is not None:
                        states[i, :, rep] = x.flatten()

                    if self.regress_input:
                        x = np.hstack((_feature.flatten(), x.flatten()))

                    # predict next value
                    pred = self.output_model.predict(x.reshape(1, -1))
                    if isinstance(pred, tuple):
                        pred, _ = pred[0], pred[1:]

                    output = self.output_scaler.unscale(pred.flatten())
                    prediction[i, rep] = float(output)

                    # transform output to new input and save it in variable _feature
                    _feature = self.input_activation(self.input_scaler.scale(output)).reshape((-1, 1))
                    _feature += errors[i]

        prediction[np.isnan(prediction) | np.isinf(prediction)] = np.nan

        # average of num_preds of all predictions
        prediction = np.nanmean(prediction, axis=-1)

        # same for the states
        if states is not None:
            states = np.mean(states, axis=-1)

        # if num_reps > 1 and simulation, set the state of the reservoir to the average state a n_predictions
        if not simulation and num_reps > 1:
            self.reservoir._state[:] = states[-1, :]

        # return values according to input
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


class OnlineReservoirModel(ReservoirModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_feature = None
        # if not hasattr(self.output_model, 'update_weight'):
        #     raise ValueError("Output model can not be trained online. It needs 'update_weight' method.")

    def _train(self):
        """
        This method can be used after new hyper_parameter shave been set.
        It only uses the instance variables (burn_in-) feature and teacher
        """

        self.update(self._burn_in_feature)

        self._fitted = self._train_and_predict(self._feature, self._teacher)
        self._errors = (self._fitted - self._teacher).flatten()
        return self._fitted

    def _train_and_predict(self, x, y):
        assert x.ndim == 1 and y.ndim == 1

        res_out = self.update(x)
        if self.regress_input:
            res_out = np.hstack((x, res_out))

        return self.output_model.fit(res_out, y)

    def predict(self, n_predictions=1, **kwargs):
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
        # reservoir = self.reservoir.copy() if simulation else self.reservoir
        feature = kwargs.get('feature', None)
        if self._last_feature is None:
            _feature = self.input_activation(self.input_scaler.scale(
                self.teacher[-1, :] if feature is None else feature))
        else:
            _feature = self._last_feature if feature is None else feature

        target = kwargs.get("target", None)

        output = np.zeros((n_predictions,), dtype=np.float64)
        rest = []
        simulation = kwargs.get('simulation', False)
        with self.reservoir.simulate(simulation):
            for i in range(n_predictions):
                # get new regressors for output model
                x = self.update(_feature)

                if self.regress_input:
                    x = np.hstack((_feature.flatten(), x.flatten()))

                # predict next value
                pred = self.output_model.predict(x.reshape(1, -1))

                if target is not None:
                    pred = self.output_model.update_weight(x, target[i])

                if isinstance(pred, tuple):
                    pred, r = pred[0], pred[1:]

                    if len(r) == 1:
                        r = r[0]

                output[i] = self.output_scaler.unscale(pred.flatten())
                # rest.append(r)
                _feature = self.input_activation(self.input_scaler.scale(pred)).reshape((-1, 1))

            if not simulation:
                self._last_feature = _feature

            return output

    # def predictor(self, simulation=True, send_new_target=True):
    #     if send_new_target:
    #         return self.predictor_with_teacher(simulation=simulation)
    #     else:
    #         return self.predictor_without_teacher(simulation=simulation)

    # def predictor_without_teacher(self, simulation=True):
    #     """
    #     let the model predict
    #     :param n_predictions: number of predictions in free run mode. This only works if
    #     number of outputs is equal to the number of inputs
    #     :param feature: features to predict
    #     :param simulation: (Boolean) if a copy of the reservoir should be used
    #     :param inject_error: (Boolean) if estimation error is added to the prediction
    #     :param num_reps: if inject_error is True, number of
    #                     repetitions performed with different errors
    #     :return: prediction generator
    #     """

    #     # reservoir = self.reservoir.copy() if simulation else self.reservoir
    #     _feature = self.input_activation(self.input_scaler.scale(self.teacher[-1, :]))

    #     with self.reservoir.simulate(simulation):
    #         try:
    #             while True:
    #                 # get new regressors for output model
    #                 x = self.update(_feature)

    #                 if self.regress_input:
    #                     x = np.hstack((_feature.flatten(), x.flatten()))

    #                 # predict next value
    #                 pred, *args = self.output_model.predict(x.reshape(1, -1))

    #                 output = self.output_scaler.unscale(pred.flatten())

    #                 yield output, x, args

    #                 # transform output to new input and save it in variable _feature
    #                 _feature = self.input_activation(self.input_scaler.scale(output)).reshape((-1, 1))
    #         except GeneratorExit:
    #             pass

    # def predictor_with_teacher(self, simulation=True):
    #     """
    #     let the model predict
    #     :param n_predictions: number of predictions in free run mode. This only works if
    #     number of outputs is equal to the number of inputs
    #     :param feature: features to predict
    #     :param simulation: (Boolean) if a copy of the reservoir should be used
    #     :param inject_error: (Boolean) if estimation error is added to the prediction
    #     :param num_reps: if inject_error is True, number of
    #                     repetitions performed with different errors
    #     :return: prediction generator
    #     """

    #     # reservoir = self.reservoir.copy() if simulation else self.reservoir
    #     _feature = self.input_activation(self.input_scaler.scale(self.teacher[-1, :]))

    #     with self.reservoir.simulate(simulation):
    #         try:
    #             while True:
    #                 # get new regressors for output model
    #                 x = self.update(_feature)

    #                 if self.regress_input:
    #                     x = np.hstack((_feature.flatten(), x.flatten()))

    #                 # predict next value
    #                 pred, *args = self.output_model.predict(x.reshape(1, -1))

    #                 output = self.output_scaler.unscale(pred.flatten())

    #                 yield output, x, args

    #                 target_output = yield
    #                 self.output_model.update_weight(x, target_output.reshape((1, -1)))

    #                 # transform output to new input and save it in variable _feature
    #                 _feature = self.input_activation(self.input_scaler.scale(output)).reshape((-1, 1))
    #         except GeneratorExit:
    #             pass
