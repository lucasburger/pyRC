#!/usr/bin/python3

from pyRC.models import deepESN
from pyRC.util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt
import json
from pyRC import util

mg = np.load("MackeyGlass_tau17.npy")

# split into train and test parts
n_pred = 1000
train_feature = mg[:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
e = deepESN()
# adds three layers according to the spetral radii
e.add_layer(size=1000, spectral_radius=[0.9, 0.95], output=[False, True])
#e.reservoir._echo = scipy.sparse.csr_matrix(util.random_echo_matrix(size=400))
r, result_dict = e.train(feature=train_feature)


# forecast
pred = e.predict(n_predictions=n_pred, simulation=True)
test_error = RMSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*n_pred), n_pred), mg[-int(1.5*n_pred):], 'b')
plt.plot(np.arange(-int(0.5*n_pred), 0), e.fitted[-int(0.5*n_pred):], 'g')
plt.plot(np.arange(-1, n_pred), np.append(e.teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])

format_hyper = {k: ["%.2f" % member for member in v] for k, v in result_dict.items()}
plt.title("Train Error = {:0.6f}, Test Error = {:0.6f}\n".format(e.error, test_error) +
          ', '.join([str(k) + ": {}".format(v) for k, v in format_hyper.items()]))
plt.show(block=True)
