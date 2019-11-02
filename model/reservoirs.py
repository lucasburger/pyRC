import numpy as np
from util import matrix_uniform, expand_echo_matrix, random_echo_matrix
from util import make_kwargs_one_length
from util import MultivariatePolynomial, MultivariateTrigoPolynomial, identity
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from copy import deepcopy


class BaseReservoir:

    def __init__(self, size=0, activation=np.tanh):
        self._state = np.zeros(shape=(size,), dtype=np.float64)
        self.activation = activation

    def copy(self):
        return deepcopy(self)

    def update(self, input_array):
        if input_array.ndim == 1 or self.size in input_array.shape and 1 in input_array.shape:
            self.state = input_array.ravel()
            return self.state
        elif self.size in input_array.shape:
            return np.apply_along_axis(self.update, axis=input_array.shape.index(self.size), arr=input_array)
        else:
            raise ValueError("Dimension mismatch of input and reservoir size")

    def _get_state(self):
        return self._state

    def _set_state(self, x):
        self._state[:] = self.activation(x)

    @property
    def state(self):
        return self._get_state()

    @state.setter
    def state(self, x):
        self._set_state(x)

    @property
    def size(self):
        return self._state.shape[0]


class ESNReservoir(BaseReservoir):

    def __init__(self, *args, bias=0.0, spectral_radius=0.95, sparsity=None, allow_sparse_echo=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias = bias
        assert spectral_radius > 0
        if sparsity is None:
            sparsity = 0.5
        self._spectral_radius = spectral_radius
        self._sparsity = sparsity
        self._echo = random_echo_matrix(size=(self.size, self.size), sparsity=sparsity, spectral_radius=spectral_radius)
        if self.size > 500 and allow_sparse_echo:
            self._echo = sparse.csr_matrix(self._echo)
        self._W_bias = matrix_uniform(self.size)
        self.__bias = self._bias * self._W_bias

    @property
    def echo(self):
        return self._echo

    def _get_bias(self):
        return self._bias

    @property
    def bias(self):
        return self._get_bias()

    def _get_spectral_radius(self):
        return self._spectral_radius

    @property
    def spectral_radius(self):
        return self._get_spectral_radius()

    def _get_sparsity(self):
        return self._sparsity

    @property
    def sparsity(self):
        return self._get_sparsity()

    def _set_state(self, x):
        super()._set_state(np.dot(self._echo, self._state) + x.flatten() + self._bias * self._W_bias)


class LeakyESNReservoir(ESNReservoir):

    def __init__(self, *args, leak=1.0, size=200, **kwargs):
        super().__init__(*args, size=size, **kwargs)
        self._leak = leak
        if sparse.issparse(self._echo):
            self._set_state = self._sparse_set_state

    def _get_leak(self):
        return self._leak

    def _set_leak(self, x):
        self._leak = min([abs(x), 1.0])

    @property
    def leak(self):
        return self._get_leak()

    @leak.setter
    def leak(self, x):
        self._set_leak(x)

    def _set_state(self, x):
        new_state = np.dot(self._echo, self._state) + x + (self._bias + self._W_bias)
        self._state[:] = self.leak * self.activation(new_state) + (1 - self.leak) * self._state
        return self._state

    def _sparse_set_state(self, x):
        new_state = self._echo * self._state + x + (self.bias * self._W_bias)
        self._state[:] = self.leak * self.activation(new_state) + (1 - self.leak) * self._state
        return self._state

    @property
    def input_size(self):
        return self.size


class TopologicalReservoir(LeakyESNReservoir):

    _layer_dict = {
        'output': True, 'input': False,
        'sparsity': 0.5, 'size': 200,
        'leak': 1.0, 'bias': 0.0}

    def __init__(self, *args, **kwargs):
        kwargs['size'] = 0
        super().__init__(*args, **kwargs)
        self.layers = dict()
        self._echo_view = self._state_view = None
        self._input_indices = np.zeros(shape=(0,), dtype=int)

    def add_layer(self, **kwargs):

        # append index to dict with key depth + 1
        size = kwargs.get('size', 200)

        new_layer = self._layer_dict.copy()
        new_layer.update(kwargs)
        self.layers[self.num_layers] = new_layer

        if kwargs.get('input', False):
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
        if layer_to is None:
            layer_to = layer_from
        if matrix is None:
            matrix = random_echo_matrix(size=(self.layers[layer_from]['size'], self.layers[layer_to]['size']), **kwargs)
        self._echo_view[layer_from, layer_to][:, :] = matrix

    def update(self, input_array):
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            new_input = np.zeros_like(self._state)
            np.put(np.zeros_like(self._state), self._input_indices, input_array)
            self.state = new_input
            return self.state
        else:
            return np.apply_along_axis(self.update, axis=1, arr=input_array)

    def __echo_view(self):
        sizes = [l['size'] for l in self.layers[0]]
        if len(set(sizes)) == 1:
            size = self.layers[0]['size']
            shape = (self.num_layers, self.num_layers, size, size)
            strides = (size * self._echo.strides[0], size * self._echo.strides[1]) + self._echo.strides
            return as_strided(self._echo, shape=shape, strides=strides)
        else:
            inds = [0] + np.cumsum(sizes).tolist()
            l = [
                [
                    self._echo[slice(i_from, j_from), slice(i_to, j_to)]
                    for i_to, j_to in zip(inds[:-1], inds[1:])
                ]
                for i_from, j_from in zip(inds[:-1], inds[1:])
            ]
            return np.array(l)

    def __state_view(self):
        sizes = [l['size'] for l in self.layers[0]]
        if len(set(sizes)) == 1:
            size = self.layers[0]['size']
            shape = (self.num_layers, size)
            strides = (size * self._state.strides[0], self._state.strides[0])
            return as_strided(self._state, shape=shape, strides=strides)
        else:
            inds = [0] + np.cumsum(sizes).tolist()
            list_of_views = list([self._state[slice(i, j)] for i, j in zip(inds[:-1], inds[1:])])
            return np.array(list_of_views)

    def copy(self):
        r = deepcopy(self)
        r._echo_view = r.__echo_view()
        r._state_view = r.__state_view()
        return r

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def input_layers(self):
        return [num for num, l in self.layers.items() if l['input']]

    @property
    def input_size(self):
        return sum(l['size'] for l in self.layers.values() if l['input'])

    @property
    def output_layers(self):
        return [num for num, l in self.layers.items() if l['output']]

    @property
    def output_size(self):
        return self.state.shape[0]

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


class ESNReservoirArray:

    def __init__(self, *args, **kwargs):
        self.reservoirs = list()
        if kwargs:
            self.add_reservoir(**kwargs)

    @make_kwargs_one_length
    def add_reservoir(self, **kwargs):
        for new_kwargs in [{k: v[i] for k, v in kwargs.items()} for i in range(max(map(len, kwargs.values())))]:
            self.reservoirs.append(LeakyESNReservoir(**new_kwargs))

        assert len(list(set(r.size for r in self.reservoirs))) == 1

    def update(self, input_array):
        return np.hstack([r.update(input_array) for r in self.reservoirs])

    def copy(self):
        return deepcopy(self)

    @property
    def size(self):
        return sum([r.size for r in self.reservoirs])

    @property
    def input_size(self):
        return self.reservoirs[0].size

    @property
    def state(self):
        return self._get_state()

    def _get_state(self):
        return np.hstack([r.state for r in self.reservoirs])


class DeepESNReservoir(ESNReservoirArray):

    def __init__(self):
        super().__init__()
        self.connections = list()
        self.input_layers = list()
        self.output_layers = list()

    @make_kwargs_one_length
    def add_layer(self, **kwargs):
        for new_kwargs in [{k: v[i] for k, v in kwargs.items()} for i in range(max(map(len, kwargs.values())))]:
            if new_kwargs.pop('input', False) or self.depth == 0:
                self.input_layers.append(self.depth)
            if new_kwargs.pop('output', True):
                self.output_layers.append(self.depth)
            if self.depth > 0:
                s1 = self.reservoirs[-1].size
                s2 = new_kwargs.get('size', 200)
                self.connections.append(random_echo_matrix(size=(s1, s2), spectral_radius=0.95))
            super().add_reservoir(**new_kwargs)

    def update(self, input_array):
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            for i, r in enumerate(self.reservoirs):
                if i in self.input_layers:
                    layer_input = input_array.copy()
                else:
                    layer_input = np.zeros_like(r.state)
                if i > 0:
                    layer_input += np.dot(self.connections[i-1], state)
                state = r.update(layer_input)
            return self.state
        else:
            return np.apply_along_axis(self.update, axis=1, arr=input_array)

    def _get_state(self):
        return np.hstack([self.reservoirs[i].state for i in self.output_layers])

    @property
    def depth(self):
        return len(self.reservoirs)

    @property
    def input_size(self):
        input_s = [self.reservoirs[i].size for i in self.input_layers]
        if len(set(input_s)) > 1:
            raise ValueError('Not the same input sizes in input connected Subreservoirs')
        return list(set(input_s))[0]

    @property
    def echo(self):
        return np.block([r.echo for r in self.reservoirs])


class SASReservoir(BaseReservoir):

    def __init__(self, *args, input_dim=1, p=1, q=1, sparsity=None, lambda_max=0.95, **kwargs):
        super().__init__(*args, activation=identity, **kwargs)

        assert 0 < lambda_max < 1
        if sparsity is None:
            sparsity = 1-float(1/self.size) if self.size > 50 else 0.9
        self.p = MultivariatePolynomial.random(shape=(self.size, self.size), input_dim=input_dim, order=p, sparsity=sparsity, spectral_radius=lambda_max)
        self.q = MultivariatePolynomial.random(shape=(self.size,), input_dim=input_dim, order=q)

    def update(self, x):
        x = x.reshape((-1, 1))
        return np.apply_along_axis(super().update, axis=1, arr=x)

    def _update_state(self, x):
        return super()._set_state(np.dot(self.p(x), self._state) + self.q(x))


class TrigoSASReservoir(SASReservoir):

    def __init__(self, *args, input_dim=1, p=1, q=1, **kwargs):
        super().__init__(*args, activation=identity, **kwargs)

        self.p = MultivariateTrigoPolynomial(self.p.coeff, self.p.input_dim, self.p.order)
        self.q = MultivariateTrigoPolynomial(self.q.coeff, self.q.input_dim, self.q.order)
