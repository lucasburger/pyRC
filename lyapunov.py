
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
e = ESN(size=200)
r, result_dict = e.train(feature=train_feature, teacher=train_teacher,
                         hyper_tuning=False, minimizer={'n_calls': 50})

pred, states = e.predict(n_predictions=num_test, inject_error=False, simulation=True)

for i in range(100):
    # forecast
    new_pred, states = e.predict(n_predictions=num_test, inject_error=False, simulation=True)

    print(np.sum(np.abs(pred - new_pred)))
    pred = new_pred
