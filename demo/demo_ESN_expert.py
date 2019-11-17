#!/usr/bin/python3

from pyRC.models import parallelESN
from pyRC.models.output_models import ExpertEvaluation
from pyRC.util import MackeyGlass, RMSE, NRMSE, MSE
import numpy as np
import matplotlib.pyplot as plt
import json


l = 4000  # length of mackey glass timeseries
mg = MackeyGlass(l).reshape((-1, 1))
mg -= np.mean(mg)  # demean

# split into train and test parts
n_pred = 1000
train_feature = mg[:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
size = 50
spectral_radius = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sizes = [size]*len(spectral_radius)

e = parallelESN(size=size, spectral_radius=spectral_radius, output_model=ExpertEvaluation(sizes=sizes))
r, result_dict = e.train(feature=train_feature, hyper_tuning=False, exclude_hyper=['leak'], error_fun=NRMSE)

# forecast
prediction_generator = e.predict(n_predictions=n_pred, simulation=True)
while True:
    try:
        errors = next(prediction_generator)
        e.output_model.update_weights((errors - test_teacher[i]).flatten())
    except StopIteration as e:
        pred = e.value
        break

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
