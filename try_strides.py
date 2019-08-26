import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def block_view(A, block= (3, 0)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)



    shape= (int(A.shape[0]/ block[0]), ) + block
    strides= (int(block[0]* A.strides[0]), 0) + A.strides

    return ast(A, shape=shape, strides=strides)


a = np.arange(16)

b = ast(a, (4, 4), (32, 8))

c = np.copy(b[0])
b[0] = np.array([100, 200, 300, 400])

print(a)
print(b)
print(c)

