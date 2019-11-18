import numpy as np
from ..util import matrix_uniform, expand_echo_matrix, random_echo_matrix
from ..util import make_kwargs_one_length, update_hyperparams
from ..util import MultivariatePolynomial, MultivariateTrigoPolynomial, identity
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from copy import deepcopy
import numba
from ..util import self_activation


class BaseReservoir:

    _saved_state = None

    def __init__(self, size=0, activation=np.tanh, random_seed=None):
        self._state = np.zeros(shape=(size,), dtype=np.float64)
        self.activation = activation

    def copy(self):
        return deepcopy(self)

    def reset(self):
        self._state[:] = 0

    def simulate(self, simulate=True):
        if simulate:
            self._saved_state = self.state.copy()
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._saved_state is not None:
            self._state = self._saved_state
            self._saved_state = None

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
        return self._state.copy()

    @property
    def state(self):
        return self._get_state()

    @state.setter
    def state(self, x):
        self._set_state(x)

    @property
    def size(self):
        return self._state.shape[0]

    @size.setter
    def size(self, size):
        pass


class ESNReservoir(BaseReservoir):

    hyper_params = {'spectral_radius': (0.0, 1.0),
                    'bias': (-1.0, 1.0)}

    def __init__(self, *args, bias=0.1, spectral_radius=0.5, sparsity=None, echo=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias = bias
        assert spectral_radius > 0

        if sparsity is None and self.size > 0:
            sparsity = max(0.0, 1-float(10/self.size))

        self._spectral_radius = spectral_radius
        self._sparsity = sparsity
        if echo is None:
            echo = random_echo_matrix(size=(self.size, self.size), sparsity=sparsity, spectral_radius=spectral_radius)
        self._echo = sparse.csr_matrix(echo)
        # self._W_bias = matrix_uniform(self.size)
        self._W_bias = np.ones((self.size,))

    @property
    def input_size(self):
        return self.size

    @property
    def echo(self):
        return self._echo

    @property
    def bias(self):
        return self._get_bias()

    def _get_bias(self):
        return self._bias

    @bias.setter
    def bias(self, other):
        self._set_bias(other)

    def _set_bias(self, other):
        self._bias = other

    @property
    def spectral_radius(self):
        return self._get_spectral_radius()

    def _get_spectral_radius(self):
        return self._spectral_radius

    @spectral_radius.setter
    def spectral_radius(self, other):
        self._set_spectral_radius(other)

    def _set_spectral_radius(self, other):
        self._echo *= other/self._spectral_radius
        self._spectral_radius = other

    def _get_sparsity(self):
        return self._sparsity

    @property
    def sparsity(self):
        return self._get_sparsity()

    def _set_state(self, x):
        super()._set_state(np.dot(self._echo, self._state) + x.flatten() + self._bias * self._W_bias)


class LeakyESNReservoir(ESNReservoir):

    hyper_params = dict(**ESNReservoir.hyper_params, **{'leak': (0.0, 1.0)})

    def __init__(self, *args, leak=1.0, size=200, **kwargs):
        super().__init__(*args, size=size, **kwargs)
        self._leak = leak

    @property
    def leak(self):
        return self._get_leak()

    def _get_leak(self):
        return self._leak

    @leak.setter
    def leak(self, other):
        self._set_leak(other)

    def _set_leak(self, x):
        self._leak = min([abs(x), 1.0])

    def _set_state(self, x):
        """
        This is using sparse matrices therefore it uses "simple" multiplication * instead of np.dot
        """
        new_state = self._echo * self._state + x + (self.bias * self._W_bias)
        self._state[:] = self.leak * self.activation(new_state) + (1.0 - self.leak) * self._state
        return self.state

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
        self.layers = list()
        #self._echo_view = self._state_view = None
        self._input_indices = np.zeros(shape=(0,), dtype=int)
        self._output_indices = np.zeros(shape=(0,), dtype=int)

    def add_layer(self, **kwargs):

        # append index to dict with key depth + 1
        size = kwargs.get('size', 200)

        new_layer = self._layer_dict.copy()
        new_layer.update(kwargs)
        self.layers.append(new_layer)

        if kwargs.get('input', False):
            self._input_indices = np.hstack((self._input_indices, np.arange(self.size, self.size+size, dtype=int)))

        if kwargs.get('output', False):
            self._output_indices = np.hstack((self._input_indices, np.arange(self.size, self.size+size, dtype=int)))

        echo_matrix_kwargs = {
            'size': size,
            'sparsity': kwargs.pop('sparsity', float(1-10/size)),
            'spectral_radius': kwargs.pop('spectral_radius', 0.95)
        }

        # create new echo and expand overall echo matrix
        self._echo = expand_echo_matrix(self._echo, new_matrix=random_echo_matrix(**echo_matrix_kwargs))

        # update spectral_radius
        self._spectral_radius = np.max(np.abs(np.linalg.eigvals(self._echo.todense())))

        # adjust state and bias
        self._state = np.hstack((self._state, np.zeros((size,))))
        self._W_bias = np.hstack((self._W_bias, matrix_uniform(size)))

        #self._echo_view = self.__echo_view()
        #self._state_view = self.__state_view()

    # def get_connection(self, layer_from, layer_to):
    #     return np.copy(self._echo_view[layer_to, layer_from])

    def set_connection(self, layer_from, layer_to=None, matrix=None, **kwargs):
        if layer_to is None:
            layer_to = layer_from

        if matrix is None:
            s1 = self.layers[layer_from]['size']
            s2 = self.layers[layer_to]['size']
            if kwargs.get('sparsity', None) is None:
                kwargs['sparsity'] = float(1-10/np.sqrt(s1*s2))

            matrix = random_echo_matrix(size=(s1, s1), **kwargs)

        sizes = [l['size'] for l in self.layers]
        inds = [0] + np.cumsum(sizes).tolist()
        echo = self._echo.todense()
        echo[inds[layer_to]:inds[layer_to+1], inds[layer_from]:inds[layer_from+1]] = matrix
        self._echo = sparse.csr_matrix(echo)
        # self._echo_view[layer_to, layer_from][:, :] = matrix

    def update(self, input_array):
        if input_array.ndim == 1 or input_array.shape[0] == 1:
            new_input = np.zeros_like(self._state)
            np.put(new_input, self._input_indices, input_array)
            self.state = new_input
            return self.state
        else:
            return np.apply_along_axis(self.update, axis=input_array.shape.index(self.input_size), arr=input_array)

    # def __echo_view(self):
    #     sizes = [l['size'] for l in self.layers]
    #     if len(set(sizes)) == 1:
    #         size = self.layers[0]['size']
    #         shape = (self.num_layers, self.num_layers, size, size)
    #         strides = (size * self._echo.strides[0], size * self._echo.strides[1]) + self._echo.strides
    #         return as_strided(self._echo, shape=shape, strides=strides)
    #     else:
    #         inds = [0] + np.cumsum(sizes).tolist()
    #         l = [
    #             [
    #                 self._echo[slice(i_from, j_from), slice(i_to, j_to)]
    #                 for i_to, j_to in zip(inds[:-1], inds[1:])
    #             ]
    #             for i_from, j_from in zip(inds[:-1], inds[1:])
    #         ]
    #         return np.array(l)

    # def __state_view(self):
    #     sizes = [l['size'] for l in self.layers]
    #     if len(set(sizes)) == 1:
    #         size = self.layers[0]['size']
    #         shape = (self.num_layers, size)
    #         strides = (size * self._state.strides[0], self._state.strides[0])
    #         return as_strided(self._state, shape=shape, strides=strides)
    #     else:
    #         inds = [0] + np.cumsum(sizes).tolist()
    #         list_of_views = list([self._state[slice(i, j)] for i, j in zip(inds[:-1], inds[1:])])
    #         return np.array(list_of_views)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def input_layers(self):
        return [num for num, l in enumerate(self.layers) if l['input']]

    @property
    def input_size(self):
        return sum(l['size'] for l in self.layers if l['input'])

    @property
    def output_layers(self):
        return [num for num, l in enumerate(self.layers) if l['output']]

    @property
    def output_size(self):
        return self.state.shape[0]

    # @property
    # def echo_view(self):
    #     return self._echo_view()

    # @property
    # def state_view(self):
    #     return self._state_view

    def _get_leak(self):
        return np.hstack([np.ones(shape=(l['size'],))*l['leak'] for l in self.layers])

    def _get_bias(self):
        return np.hstack([np.ones(shape=(l['size'],))*l['bias'] for l in self.layers])

    def _get_state(self):
        return self._state[self._output_indices].copy()
        # return np.hstack(self._state_view[self.output_layers])


class DeepESNReservoirTry(TopologicalReservoir):

    def __init__(self):
        super().__init__()

    @make_kwargs_one_length
    def add_layer(self, **kwargs):
        """
        Example add_layer(spectral_radius=[0.9, 0.95, 0.99])
        """
        if self.num_layers == 0:
            kwargs['input'] = True
        elif self.num_layers > 0 and kwargs.get('input', None) is None:
            kwargs['input'] = False

        super().add_layer(**kwargs)
        if self.num_layers > 1:
            super().set_connection(self.num_layers-2, self.num_layers-1, spectral_radius=0.95)


class ESNReservoirArray:

    hyper_params = {}
    _fixed_size = None
    _saved_state = None
    _reservoir_class = LeakyESNReservoir

    def __init__(self, *args, **kwargs):
        self.reservoirs = list()
        if kwargs:
            self.add_reservoir(**kwargs)
            self.fixed_size = self.reservoirs[-1].size

    @make_kwargs_one_length
    def add_reservoir(self, **kwargs):
        new_res = self._reservoir_class(**kwargs)
        self.reservoirs.append(new_res)
        self.hyper_params = update_hyperparams(self.hyper_params, new_res.hyper_params)

    def update(self, input_array):
        return np.hstack([r.update(input_array) for r in self.reservoirs])

    def reset(self):
        for r in self.reservoirs:
            r.reset()

    def simulate(self, simulate=True):
        if simulate:
            self._saved_state = [r._state for r in self.reservoirs]
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._saved_state:
            for i, r in enumerate(self.reservoirs):
                r._state[:] = self._saved_state[i]
            self._saved_state = None

    @property
    def size(self):
        return sum([r.size for r in self.reservoirs])

    @size.setter
    def size(self, size):
        s = int(size/len(self.reservoirs))
        for r in self.reservoirs:
            r.size = s

    @property
    def input_size(self):
        return self._fixed_size

    @property
    def state(self):
        return self._get_state()

    def _get_state(self):
        return np.hstack([r.state for r in self.reservoirs])

    @property
    def spectral_radius(self):
        return [r.spectral_radius for r in self.reservoirs]

    @spectral_radius.setter
    def spectral_radius(self, other):
        if isinstance(other, float):
            for r in self.reservoirs:
                r.spectral_radius = other
        else:
            for i, sr in enumerate(other):
                self.reservoirs[i].spectral_radius = sr

    @property
    def bias(self):
        return [r.bias for r in self.reservoirs]

    @bias.setter
    def bias(self, other):
        if isinstance(other, float):
            for r in self.reservoirs:
                r.bias = other
        else:
            for i, b in enumerate(other):
                self.reservoirs[i].bias = b

    @property
    def leak(self):
        return [r.leak for r in self.reservoirs]

    @leak.setter
    def leak(self, other):
        if isinstance(other, float):
            for r in self.reservoirs:
                r.leak = other
        else:
            for i, l in enumerate(other):
                self.reservoirs[i].leak = l

    @property
    def fixed_size(self):
        return self._fixed_size

    @fixed_size.setter
    def fixed_size(self, other):
        if self._fixed_size is None:
            self._fixed_size = other
        else:
            raise ValueError(f"fixed size for reservoir array already set {self._fixed_size}")


class SASReservoir(BaseReservoir):

    hyper_params = {'spectral_radius': (0.0, 1.0)}

    def __init__(self, *args, input_dim=1, order_p=2, order_q=2, sparsity=None, spectral_radius=0.95, **kwargs):

        if kwargs.get('size', 0) == 0:
            kwargs['size'] = 200

        if kwargs.get('activation', None) is None:
            kwargs['activation'] = self_activation()

        super().__init__(*args, **kwargs)
        self.input_dim = input_dim

        # assert 0 < spectral_radius < 1
        if sparsity is None:
            sparsity = min(0.0, 1-float(10/self.size))

        self.p = MultivariatePolynomial.random(shape=(self.size, self.size), input_dim=input_dim, order=order_p,
                                               sparsity=sparsity, spectral_radius=spectral_radius, matrix_type='full')
        self.q = MultivariatePolynomial.random(shape=(self.size,), input_dim=input_dim, order=order_q)

    def update(self, x):
        return np.apply_along_axis(super().update, axis=x.shape.index(self.input_size), arr=x)

    def _set_state(self, x):
        return super()._set_state(np.dot(self.p(x), self._state) + self.q(x))

    @property
    def input_size(self):
        return self.input_dim

    @property
    def spectral_radius(self):
        return self.p.spectral_radius

    @spectral_radius.setter
    def spectral_radius(self, other):
        self.p.spectral_radius = other


class TrigoSASReservoir(SASReservoir):

    def __init__(self, *args, input_dim=2, p=2, q=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.p = MultivariateTrigoPolynomial(self.p.input_dim, self.p.order, self.p.coeff)
        self.q = MultivariateTrigoPolynomial(self.q.input_dim, self.q.order, self.q.coeff)


class DeepESNReservoir(ESNReservoirArray):

    def __init__(self):
        super().__init__()
        self.connections = list()
        self.input_layers = list()
        self.output_layers = list()

    @make_kwargs_one_length
    def add_layer(self, **kwargs):
        """
        Example add_layer(spectral_radius=[0.9, 0.95, 0.99])
        """
        if kwargs.pop('input', False) or self.depth == 0:
            self.input_layers.append(self.depth)
        if kwargs.pop('output', True):
            self.output_layers.append(self.depth)
        if self.depth > 0:
            s1 = self.reservoirs[-1].size
            s2 = kwargs.get('size', 200)
            self.connections.append(random_echo_matrix(size=(s1, s2), spectral_radius=0.95))
        super().add_reservoir(**kwargs)

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
