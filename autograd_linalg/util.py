import autograd.numpy as np

T = lambda x: np.swapaxes(x, -1, -2)
symm = lambda x: (x + T(x)) / 2.
C = lambda x: np.require(x, x.dtype, 'C')
