from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import multigammaln
from autograd import grad
from autograd.util import make_tuple

import mniw

# special case of mniw

al2d = np.atleast_2d
add_dims = lambda A, b, c, d: (A, b[:,None], al2d(c), d)
remove_dims = lambda A, B, C, d: make_tuple(A, B.ravel(), C[0,0], d)

def standard_to_natural(nu, S, m, kappa):
    A, B, C, d = mniw.standard_to_natural(nu, S, m[:,None], al2d(kappa))
    return C, B.ravel(), A[0,0], d

def expectedstats(natparam):
    return remove_dims(*mniw.expectedstats(add_dims(*natparam)))

def logZ(natparam):
    return mniw.logZ(add_dims(*natparam))

def natural_sample(natparam):
    A, Sigma = mniw.natural_sample(add_dims(*natparam))
    return np.ravel(A), Sigma
