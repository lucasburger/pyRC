# pyRC

# Preliminaries

This library was written in Python 3 and contains codes, that is not compatible with Python 2.7. However, changing a few lines and import statements will make it work with both versions.

## Installation

From source (using git):

````
git clone https://github.com/lucasburger/pyRC.git
cd pyRC
pip3 install -r requirements.txt
````

## The Library

### Echo State Networks
- (Leaky-Integrator) ESN
- Deep ESN [Gallicchio and Micheli (2017)](https://arxiv.org/abs/1712.04323)
- (Arbitrary) Topological ESN
- Parallel ESN

### State Affine Systems

[Grigoryeva and Ortega (2018)](https://arxiv.org/pdf/1712.00754.pdf)
- Polynomial SAS
- Trigonometric SAS

### Output Models

Output models can basically be chosen from any [scikit-learn](https://scikit-learn.org/stable/) model having a fit and predict method. Alternatively, they can be specified separately and, for convenience, can derive from model.output_models.BaseOutputModel:

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
### Hyperparameter Optimization

Although, Reservoir Computers are lightweight in terms of training, the optimization of hyperparameters - as with any machine learning algorithm - bares some challenge.
Automatic optimization is performed using the library [scikit-optimize](https://scikit-optimize.github.io) which is unfortunately not maintained anymore.
You can let the optimizer choose the hyperparameters (i.e. spectral radius, bias or leak) by setting the ```tune_hyper=True```when calling the train method.

## Examples

This is a very basic example of training a leaky-integrator ESN on the [Mackey-Glass chaotic timeseries](http://www.scholarpedia.org/article/Mackey-Glass_equation) with common parameter choice $\tau = 17$.

```python
from model.EchoStateNetwork import EchoStateNetwork as ESN
from util import MackeyGlass, RMSE
import numpy as np
import matplotlib.pyplot as plt

l = 2000  # length of mackey glass timeseries
mg = MackeyGlass(l, drop_out=0.1).reshape((-1, 1))
mg -= np.mean(mg)  # demean

# split into burn_in, train and test parts
num_test = 500
train_feature = mg[:-num_test-1]
train_teacher = mg[1:-num_test]
test_teacher = mg[-num_test:]


# set up ESN and train
e = ESN()
r, result_dict = e.train(feature=train_feature, teacher=train_teacher, hyper_tuning=False)

# forecast
pred = e.predict(n_predictions=num_test, simulation=True)
test_error = RMSE(pred, test_teacher)

# visualize results
fig = plt.figure(1)
plt.plot(np.arange(-int(0.5*num_test), num_test), mg[-int(1.5*num_test):], 'b')
plt.plot(np.arange(-int(0.5*num_test), 0), e.fitted[-int(0.5*num_test):], 'g')
plt.plot(np.arange(-1, num_test), np.append(train_teacher[-1], pred), 'r')
plt.legend(['target', 'fitted', 'forecast'])
plt.title(f"Train Error = {e.error:0.6f}, Test Error = {test_error:0.6f}\n"+str(result_dict))
plt.show(block=True)
```


