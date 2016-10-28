import numpy as np
import autograd.numpy as anp
from autograd.core import primitive
from autograd.scipy.linalg import _flip

import cython_linalg as cyla
from util import T, symm, C

### solve_triangular

@primitive
def solve_triangular(a, b, trans=0, lower=False, **kwargs):
    '''Just like scipy.linalg.solve_triangular on real arrays, except this
    function broadcasts over leading dimensions like np.linalg.solve.'''
    flat_a = np.reshape(a, (-1,) + a.shape[-2:])
    flat_b = np.reshape(b, flat_a.shape[:-1] + (-1,))
    flat_result = cyla.solve_triangular(C(flat_a), C(flat_b),
                                        trans=trans, lower=lower)
    return np.reshape(flat_result, b.shape)

def make_grad_solve_triangular(ans, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if (lower ^ (_flip(a, trans) == 'N')) else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else T(x)
    ans = anp.reshape(ans, a.shape[:-1] + (-1,))

    def solve_triangular_grad(g):
        v = solve_triangular(a, g, trans=_flip(a, trans), lower=lower)
        return -transpose(tri(anp.matmul(anp.reshape(v, ans.shape), T(ans))))

    return solve_triangular_grad
solve_triangular.defgrad(make_grad_solve_triangular)
solve_triangular.defgrad(lambda ans, a, b, trans=0, lower=False, **kwargs:
                         lambda g: solve_triangular(a, g, trans=_flip(a, trans), lower=lower),
                         argnum=1)

### cholesky

solve_trans = lambda L, X: solve_triangular(L, X, lower=True, trans='T')
solve_conj = lambda L, X: solve_trans(L, T(solve_trans(L, T(X))))
phi = lambda X: anp.tril(X) / (1. + anp.eye(X.shape[-1]))

cholesky = primitive(np.linalg.cholesky)
cholesky.defgrad(lambda L, A: lambda g: symm(solve_conj(L, phi(anp.matmul(T(L), g)))))
