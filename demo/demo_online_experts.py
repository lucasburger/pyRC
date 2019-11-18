#!/usr/bin/python3

from pyRC.models import parallelESN
from pyRC.models.output_models_online import ExpertEvaluation
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

sizes = [200] * 5
# set up ESN and train
e = parallelESN(size=sizes, bias=0.2, spectral_radius=[0.9, 0.8, 0.85, 0.9, 0.9], output_model=ExpertEvaluation(sizes=sizes))
r, result_dict = e.train(feature=train_feature, hyper_tuning=False, exclude_hyper=['leak'], error_fun=NRMSE)

pred_offline = e.predict(n_pred)
test_error = MSE(pred_offline, test_teacher)

pred_online = np.zeros((n_pred, ), dtype=np.float64)

with e.reservoir.simulate():
    for i in range(test_teacher.shape[0]):
        pred_online[i], experts = e.predict(1)
        errors = experts.flatten() - test_teacher[i, :].flatten()
        _ = e.output_model.update_weight(errors)

print(pred_online[-5:])
# predictor.close()

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*n_pred), n_pred), mg[-int(1.5*n_pred):], 'b')
plt.plot(np.arange(-int(0.5*n_pred), 0), e.fitted[-int(0.5*n_pred):], 'g')
plt.plot(np.arange(-1, n_pred), np.append(e.teacher[-1], pred_offline), 'r')
plt.plot(np.arange(-1, n_pred), np.append(e.teacher[-1], pred_online), 'm')
plt.legend(['target', 'fitted', 'forecast_offline', 'forecast_online'])
plt.title("Train Error = {:0.6f}, Test Error = {:0.6f}\n".format(e.error, test_error) +
          ', '.join([str(k) + ": {:1.4f}".format(v) for k, v in result_dict.items()]))
plt.show(block=True)
