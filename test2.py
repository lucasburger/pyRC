
from model.reservoirsV2 import Reservoir, LeakyReservoir, DeepReservoir
import numpy as np
import util

size = 100
activation = util.identity
leak = 1.0
bias = 0.0
input_array = np.arange(size*10).reshape((10, size))

echo = np.eye(size)


r1 = Reservoir(size=size, activation=activation, bias=bias)
r1._echo = echo
r2 = LeakyReservoir(size=size, activation=activation, leak=leak, bias=bias)
r2._echo = echo
r3 = DeepReservoir(activation=activation)
r3.add_layer(size=size, leak=leak, bias=bias, input=True)
r3.set_connection(0, 0, matrix=echo)
r1.update(input_array)
r2.update(input_array)
r3.update(input_array)

print(r1.state[:10])
print(r2.state[:10])
print(r3.state[:10])