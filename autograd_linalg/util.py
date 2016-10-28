import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import check_grads

T = lambda x: np.swapaxes(x, -1, -2)
symm = lambda x: (x + T(x)) / 2.
C = lambda x: np.require(x, x.dtype, 'C')

def check_symmetric_matrix_grads(fun, *args):
    symmetrize = lambda A: symm(np.tril(A))
    new_fun = lambda *args: fun(symmetrize(args[0]), *args[1:])
    return check_grads(new_fun, *args)

def rand_psd(D):
    mat = npr.randn(D,D)
    return np.dot(mat, mat.T) + 5 * np.eye(D)
