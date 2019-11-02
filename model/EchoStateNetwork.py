
from copy import copy
import numpy as np
from model.BaseModel import ReservoirModel
from model.output_models import RidgeRegressionCV
from model.reservoirs import LeakyESNReservoir, DeepESNReservoir, ESNReservoirArray
from model.scaler import tanh, identity
import util


class EchoStateNetwork(ReservoirModel):

    _reservoir_class = LeakyESNReservoir

    def __repr__(self):
        return f"(ESN: N={self.size}, SR={self.spectral_radius})"


class deepESN(EchoStateNetwork):
    _reservoir_class = DeepESNReservoir

    def add_layer(self, *args, **kwargs):
        self.reservoir.add_layer(*args, **kwargs)

    @property
    def layers(self):
        return getattr(self.reservoir, 'layers', None)


class parallelESN(EchoStateNetwork):
    _reservoir_class = ESNReservoirArray

    def add_reservoir(self, *args, **kwargs):
        self.reservoir.add_reservoir(*args, **kwargs)

    @property
    def reservoirs(self):
        return getattr(self.reservoir, 'reservoir', None)


class MultiStepESN(EchoStateNetwork):

    _reservoir_class = ESNReservoirArray

    def __init__(self, time_steps=None, **kwargs):

        if kwargs is None and time_steps is None:
            time_steps = 1

        self.time_steps = time_steps

        # cast all kwargs into lists
        kwargs = {k: [v] if not isinstance(
            v, list) else v for k, v in kwargs.items()}

        if all(len(k) == 1 for k in kwargs):
            self.networks = [EchoStateNetwork(
                **kwargs) for _ in range(time_steps)]
        else:
            try:
                kwargs = [{k: v[i] if len(v) > 1 else v[0] for k, v in kwargs.items()} for i in range(time_steps)]
                self.networks = [EchoStateNetwork(**k) for k in kwargs]
            except:
                raise ValueError("Wrong combination of time_steps and ESN parameters.")

    def train(self, feature, burn_in_feature=None, burn_in_split=0.1, teacher=None):

        if burn_in_feature is None:
            burn_in_ind = int(burn_in_split * feature.shape[0])
            burn_in_feature = feature[:burn_in_ind, :]
            feature = feature[burn_in_ind:, :]

        if teacher is None:
            x = feature.ravel()
            window = self.time_steps + 1
            shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
            strides = x.strides + (x.strides[-1],)
            x = np.lib.stride_tricks.as_strided(
                x, shape=shape, strides=strides)
            feature = x[:, 0]
            teacher = x[:, 1:]

        return [e.train(burn_in_feature=burn_in_feature, feature=feature, teacher=teacher[:, i])
                for i, e in enumerate(self.networks)]

    def predict(self, n_predictions=1, **kwargs):
        return np.hstack([esn.predict(n_predictions=1, **kwargs) for esn in self.networks])
