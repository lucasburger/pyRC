from model.reservoirs import ReservoirV2
from util import expand_echo_matrix
import numpy as np
import datetime as dt


r = ReservoirV2()

r.add_layer(size=200, leak=0.1, input_connection=True, output_connection=True)
r.add_layer(size=200, leak=0.1, input_connection=False, output_connection=True)
r.add_layer(size=200, leak=0.1, input_connection=False, output_connection=True)
r.add_layer(size=200, leak=0.1, input_connection=False, output_connection=True)

r.update(np.eye(20))
