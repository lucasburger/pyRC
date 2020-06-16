# pyRC

## Preliminaries

This library was written in Python 3 and contains codes, that is not compatible with Python 2.7. However, changing a few lines and import statements will make it work with both versions.

Documentation is "not very good" to "almost not existent". This will come in later stages.

## Installation

With pip and git from this GitHub repo:

```
pip install git+https://github.com/lucasburger/pyRC.git
```

## The Library

### Echo State Networks
- (Leaky-Integrator) ESN
- Deep ESN [Gallicchio and Micheli (2017)](https://arxiv.org/abs/1712.04323)
- Parallel ESN (work in progress)
- (Arbitrary) Topological ESN (work in progress)

### State Affine Systems (Work in progress, still buggy)

[Grigoryeva and Ortega (2018)](https://arxiv.org/pdf/1712.00754.pdf)
- Polynomial SAS
- Trigonometric SAS

### Output Models

Output models can basically be chosen from any [scikit-learn](https://scikit-learn.org/stable/) model having a fit and predict method. In case, the model performs cross validation, make sure to store the cv-scores. 
Alternatively, they can be specified separately and, for convenience, can derive from model.output_models.BaseOutputModel:

### Hyperparameter Optimization

Although, Reservoir Computers are lightweight in terms of training, the optimization of hyperparameters - as with any machine learning algorithm - bares some challenge.
Automatic optimization is performed using the library [scikit-optimize](https://scikit-optimize.github.io) which is unfortunately not maintained anymore.
You can let the optimizer choose the hyperparameters (i.e. spectral radius, bias or leak) by setting the ```hyper_tuning=True```when calling the train method.

### Examples

This is a very basic example of training a ESN on the [Mackey-Glass chaotic timeseries](http://www.scholarpedia.org/article/Mackey-Glass_equation).

```python
from model.EchoStateNetwork import EchoStateNetwork as ESN
from util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt

l = 2000  # length of mackey glass timeseries
mg = MackeyGlass(l, drop_out=0.1).reshape((-1, 1))
mg -= np.mean(mg)  # demean

# split into train and test parts
n_pred = 500
train_feature = mg[:-n_pred-1]
train_teacher = mg[1:-n_pred]
test_teacher = mg[-n_pred:]

# set up ESN and train
e = ESN(leak=1.0)
r, result_dict = e.train(feature=train_feature, teacher=train_teacher, hyper_tuning=True, exclude_hyper=['leak'])

# forecast
pred = e.predict(n_predictions=n_pred, simulation=True)
test_error = RMSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*n_pred), n_pred), mg[-int(1.5*n_pred):], 'b')
plt.plot(np.arange(-int(0.5*n_pred), 0), e.fitted[-int(0.5*n_pred):], 'g')
plt.plot(np.arange(-1, n_pred), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title("Train Error = {:0.6f}, Test Error = {:0.6f}\n".format(e.error, test_error) + \
    ', '.join([str(k) + ": {:1.4f}".format(v) for k, v in result_dict.items()]))
plt.show(block=True)
```


## Basic Usage

### Output Models

To write your own output_models, your own classes can inherit from pyRC.output_models.BaseOutputModel for convenience.

 ```python
from abc import ABCMeta, abstractmethod, abstractproperty


class BaseOutputModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        """
        This function must be overridden by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        This function must be overridden by subclasses.
        """
        pass

    @abstractproperty
    def error(self):
        pass

```