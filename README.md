# pyRC


## The Library
This Reservoir Computing library includes:

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

Output models can basically be chosen from any can be specified separately and have to have the following form:

`from abc import ABCMeta, abstractmethod


class BaseOutputModel:
    __metaclass__ = ABCMeta
    error = None

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
`


## Examples

`This is code`


