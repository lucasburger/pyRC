
from model.EchoStateNetwork import EchoStateNetwork as ESN
from util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt
import json

np.random.seed(42)

l = 2000  # length of mackey glass timeseries
mg = MackeyGlass(l, drop_out=0.1)
mg -= np.mean(mg)  # demean
mg = mg.reshape((-1, 1))  # reshape to column vector

# split into burn_in, train and test parts
num_test = 500
train_feature = mg[:-num_test-1]
train_teacher = mg[1:-num_test]
test_teacher = mg[-num_test:]


# set up ESN and train
e = ESN()
r, result_dict = e.train(feature=train_feature, teacher=train_teacher, hyper_tuning=False)

# forecast
pred = e.predict(n_predictions=num_test, inject_error=False, simulation=True)
test_error = RMSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*num_test), num_test), mg[-int(1.5*num_test):], 'b')
plt.plot(np.arange(-int(0.5*num_test), 0), e.fitted[-int(0.5*num_test):], 'g')
plt.plot(np.arange(-1, num_test), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title(f"Train Error = {e.error:0.6f}, Test Error = {test_error:0.6f}")
plt.show(block=True)
