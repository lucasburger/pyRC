#!/usr/bin/python3

from pyRC.models.EchoStateNetwork import EchoStateNetwork as ESN
from pyRC.util import MackeyGlass, NRMSE, lyapunov_exponent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mg = np.load("MackeyGlass_tau17.npy")

# split into train and test parts
n_pred = 1000
train_feature = mg[:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
e = ESN(size=1000, bias=0.2, spectral_radius=0.95)
r, result_dict = e.train(feature=train_feature, burn_in_split=1000)

LE = lyapunov_exponent(e)

print(f"LE: {LE:1.6f}")
