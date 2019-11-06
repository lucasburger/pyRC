#!/usr/bin/python3

from model.EchoStateNetwork import EchoStateNetwork as ESN
from util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt
import json

l = 2000  # length of mackey glass timeseries
mg = MackeyGlass(l, drop_out=0.1).reshape((-1, 1))
mg -= np.mean(mg)  # demean
mg = mg  # reshape to column vector

# split into burn_in, train and test parts
num_test = 500
train_feature = mg[:-num_test-1]
train_teacher = mg[1:-num_test]
test_teacher = mg[-num_test:]


# set up ESN and train
e = ESN()
r, result_dict = e.train(feature=train_feature, teacher=train_teacher, hyper_tuning=True)

# forecast
pred, states = e.predict(n_predictions=num_test, inject_error=False, simulation=False, return_states=True)
test_error = RMSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*num_test), num_test), mg[-int(1.5*num_test):], 'b')
plt.plot(np.arange(-int(0.5*num_test), 0), e.fitted[-int(0.5*num_test):], 'g')
plt.plot(np.arange(-1, num_test), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title("Train Error = {:0.6f}, Test Error = {:0.6f}\n".format(e.error, test_error) + ', '.join([str(k) + ": {:1.4f}".format(v) for k, v in result_dict.items()]))
plt.show(block=True)
