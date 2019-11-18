#!/usr/bin/python3

from pyRC.models.EchoStateNetwork import EchoStateNetwork as ESN
from pyRC.util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy
from tqdm import tqdm

from scipy.io import loadmat

import cProfile

l = 5000  # length of mackey glass timeseries
initial_condition = np.hstack((np.ones((1,))*1.2, np.zeros((16,))))
mg = MackeyGlass(l, drop_out=0.1, random_seed=42).reshape((-1, 1))
mg -= np.mean(mg)  # demean

# split into train and test parts
n_pred = 1000
train_feature = mg[:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
e = ESN(size=1000, bias=0.2, spectral_radius=0.95)
r, result_dict = e.train(feature=train_feature, burn_in_split=1000, error_fun=RMSE)

error_size = 1e-7

pred = e.predict(317)
org_state = e.reservoir._state.copy()


LEs = []
np.random.seed(42)
for _ in tqdm(range(100)):
    e.reservoir._state = org_state + np.random.uniform(low=-error_size, high=error_size, size=(1000,))

    pred_pert = e.predict(317, simulation=True)

    d1 = np.sqrt(np.sum((pred[100:117] - pred_pert[100:117])**2))
    d2 = np.sqrt(np.sum((pred[300:317] - pred_pert[300:317])**2))
    LE = np.log(d2/d1)/200

    LEs.append(LE)

LE = np.mean(LEs)
print(f"LE: {LE:1.6f}")
