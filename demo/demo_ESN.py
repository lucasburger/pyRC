#!/usr/bin/python3

from pyRC.models import EchoStateNetwork as ESN
from pyRC.util import MackeyGlass, RMSE, NRMSE, MSE, log_activation
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mg = loadmat('MGTimeseries_dde23.mat')['MGseries_dde23_tau17'].T

# split into train and test parts
n_pred = 1000
train_feature = mg[:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
e = ESN(size=1000, bias=0.2, spectral_radius=0.95)
r, result_dict = e.train(feature=train_feature)

# forecast
pred = e.predict(n_predictions=n_pred, simulation=True)
test_error = MSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*n_pred), n_pred), mg[-int(1.5*n_pred):], 'b')
plt.plot(np.arange(-int(0.5*n_pred), 0), e.fitted[-int(0.5*n_pred):], 'g')
plt.plot(np.arange(-1, n_pred), np.append(e.teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title("Train Error = {:0.6f}, Test Error = {:0.6f}\n".format(e.error, test_error) +
          ', '.join([str(k) + ": {:1.4f}".format(v) for k, v in result_dict.items()]))
plt.show(block=True)
