import numpy as np
from util import matrix_uniform, expand_echo_matrix, random_echo_matrix
from numpy.lib.stride_tricks import as_strided
from copy import deepcopy


class BaseReservoir:

    def __init__(self, size=0, activation=np.tanh):
        self._state = np.zeros(shape=(size,), dtype=np.float64)
        self._echo = np.zeros(shape=(size, size), dtype=np.float64)
        self.activation = activation

    def copy(self):
        return deepcopy(self)

    @property
    def state(self):
        return self._state

    @property
    def echo(self):
        return self._echo

    @property
    def size(self):
        return self._state.shape[0] 


class Reservoir:

    def __init__(self, size=200, activation=np.tanh, sparsity=None, spectral_radius=0.9, bias=0.0, leak=1.0):
        self.activation = activation
        self._spectral_radius = spectral_radius
        self._bias = bias
        self._leak = leak
        if sparsity is None: sparsity = 1.0 - float(10.0/size)
        self._sparsity = sparsity
        self._state = np.zeros((size,), dtype=np.float64)
        self._echo = random_echo_matrix(size=size, sparsity=sparsity, spectral_radius=spectral_radius)
        self._W_bias = matrix_uniform(size)

    def update(self, input_array):
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            self.state = input_array.flatten()
            return self.state
        else:
            return np.apply_along_axis(self.update, axis=1, arr=input_array)

    def copy(self):
        return deepcopy(self)

    @property
    def size(self):
        return self._state.shape[0]

    @property
    def state(self):
        return self._state

    @property
    def echo(self):
        return self._echo

    @property
    def sparsity(self):
        return self._sparsity
    
    @property
    def spectral_radius(self):
        return self._spectral_radius

    @spectral_radius.setter
    def spectral_radius(self, sr):
        if self._echo is not None:
            self._echo *= (sr/self._spectral_radius)
            self._spectral_radius = sr

    @state.setter
    def state(self, x):
        """
        calculates the new state using the recurrent part, the input and the bias.
        Updates the state using leak, new state and old state
        :param x: new input
        """
        new_state = np.dot(self._echo, self._state) + x.flatten() + self._bias * self._W_bias
        self._state[:] = self._leak * self.activation(new_state) + (1 - self._leak) * self._state

    @property
    def input_size(self):
        return self.size


class DeepReservoir(BaseReservoir):

    echo = property(BaseReservoir.echo.__get__)
    size = property(BaseReservoir.size.__get__)

    _layer_dict = {
        'output': True, 'input': False, 
        'sparsity': 0.0, 'size': 200,
        'leak': 1.0, 'bias': 0.0}

    def __init__(self, *args, activation=np.tanh, **kwargs):
        self.layers, self.input_layers, self.output_layers = dict(), list(), list()
        self.input_size = self.output_size = self.num_layers = 0
        self._echo_view = self._state_view = None
        self._input_indices = np.zeros(shape=(0,), dtype=int)
        self._W_bias = np.zeros((0,), dtype=np.float64)
        super().__init__(size=0, activation=activation)

    def add_layer(self, **kwargs):

        # append index to dict with key depth + 1 
        size = kwargs.get('size', 200)
        self.num_layers += 1

        new_layer = self._layer_dict.copy()
        new_layer.update(kwargs)
        self.layers[self.num_layers - 1] = new_layer

        if kwargs.get('output', True): 
            self.output_size += size
            self.output_layers.append(self.num_layers - 1)
        if kwargs.get('input', False): 
            self.input_size += size
            self.input_layers.append(self.num_layers - 1)
            self._input_indices = np.hstack((self._input_indices, np.arange(self.size, self.size+size, dtype=int)))

        echo_matrix_kwargs = {
            'size': size,
            'sparsity': kwargs.get('sparsity', 0.0),
            'spectral_radius': kwargs.get('spectral_radius', 0.9)
        }

        # create new echo and expand overall echo matrix
        self._echo = expand_echo_matrix(self._echo, new_matrix=random_echo_matrix(**echo_matrix_kwargs))

        # set spectral_radius
        self._spectral_radius = np.max(np.abs(np.linalg.eigvals(self._echo)))

        # adjust state and bias
        self._state = np.hstack((self._state, np.zeros((size,))))
        self._W_bias = np.hstack((self._W_bias, matrix_uniform(size)))

        self._echo_view = self.__echo_view()
        self._state_view = self.__state_view()

    def get_connection(self, layer_from, layer_to):
        return np.copy(self._echo_view[layer_from, layer_to])

    def set_connection(self, layer_from, layer_to=None, matrix=None, **kwargs):
        if layer_to is None: layer_to = layer_from
        if matrix is None: matrix = random_echo_matrix(size=(self.layers[0]['size'],)*2, **kwargs)
        self._echo_view[layer_from, layer_to] = matrix

    def update(self, input_array):
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            new_input = np.zeros_like(self._state)
            new_input[self._input_indices] = input_array
            self.state = new_input
            return self.state
        else:
            return np.apply_along_axis(self.update, axis=1, arr=input_array)

    def __echo_view(self):
        size = self.layers[0]['size']
        shape = (self.num_layers, self.num_layers, size, size)
        strides = ( size * self._echo.strides[0] , size * self._echo.strides[1]) + self._echo.strides
        return as_strided(self._echo, shape=shape, strides=strides)

    def __state_view(self):
        size = self.layers[0]['size']
        shape = (self.num_layers, size)
        strides = (size * self._state.strides[0], self._state.strides[0])
        return as_strided(self._state, shape=shape, strides=strides)

    def copy(self):
        r = deepcopy(self)
        r._echo_view = r.__echo_view()
        r._state_view = r.__state_view()
        return r

    @property
    def echo_view(self):
        return self._echo_view

    @property
    def state_view(self):
        return self._state_view

    def _get_leak(self):
        return np.hstack([np.ones(shape=(l['size'],))*l['leak'] for l in self.layers.values()])
    
    def _get_bias(self):
        return np.hstack([np.ones(shape=(l['size'],))*l['bias'] for l in self.layers.values()])

    def _get_state(self):
        return np.hstack(self._state_view[self.output_layers])

    @property
    def hidden_state(self):
        return self._state
