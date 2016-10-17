from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import multigammaln, digamma
from autograd import grad
from autograd.util import make_tuple

from svae.util import symmetrize, outer
from gaussian import pack_dense, unpack_dense
import mniw  # niw is a special case of mniw

# NOTE: can compute Cholesky then avoid the other cubic computations,
# but numpy/scipy has no dpotri or solve_triangular that broadcasts
# NOTE: can increase parallelism in some of these formulas (with repeated work)

def expectedstats(natparam, fudge=1e-8):
    S, m, kappa, nu = natural_to_standard(natparam)
    d = m.shape[-1]

    E_J = nu[...,None,None] * symmetrize(np.linalg.inv(S)) + fudge * np.eye(d)
    E_h = np.matmul(E_J, m[...,None])[...,0]
    E_hTJinvh = d/kappa + np.matmul(m[...,None,:], E_h[...,None])[...,0,0]
    E_logdetJ = (np.sum(digamma((nu[:,None] - np.arange(d)[None,:])/2.)) \
                 + d*np.log(2.)) - np.linalg.slogdet(S)[1]

    return pack_dense((-1./2 * E_J, E_h, -1./2 * E_hTJinvh, 1./2 * E_logdetJ))

def logZ(natparam):
    S, m, kappa, nu = natural_to_standard(natparam)
    d = m.shape[-1]
    return np.sum(d*nu/2.*np.log(2.) + multigammaln(nu/2., d)
                  - nu/2.*np.linalg.slogdet(S)[1] + d/2.*np.log(kappa))

def natural_to_standard(natparam):
    A, b, kappa, nu = unpack_dense(natparam)
    m = b / kappa[:,None]
    S = A - outer(b, m)
    return S, m, kappa, nu

def standard_to_natural(S, m, kappa, nu):
    A = S + kappa * outer(m, m)
    b = kappa * m
    return pack_dense((A, b, kappa, nu))
