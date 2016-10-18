from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from functools import partial

from svae.util import T

# NOTE: can compute Cholesky then avoid the other cubic computations,
# but numpy/scipy has no dpotri or solve_triangular that broadcasts

def expectedstats(natparam):
    neghalfJ, h, _, _ = unpack_dense(natparam)
    J = -2*neghalfJ
    Ex = np.linalg.solve(J, h)
    ExxT = np.linalg.inv(J) + Ex[...,None] * Ex[...,None,:]
    En = np.ones(J.shape[0]) if J.ndim == 3 else 1.
    return pack_dense((ExxT, Ex, En, En))

def logZ(natparam):
    neghalfJ, h, a, b = unpack_dense(natparam)
    J = -2*neghalfJ
    L = np.linalg.cholesky(J)
    return 1./2 * np.sum(h * np.linalg.solve(J, h)) \
        - np.sum(np.log(np.diagonal(L, axis1=-1, axis2=-2))) \
        - np.sum(a + b)

def natural_sample(natparam, num_samples):
   neghalfJ, h, _, _ = unpack_dense(natparam)
   sample_shape = np.shape(h) + (num_samples,)
   J = -2*neghalfJ
   L = np.linalg.cholesky(J)
   noise = np.linalg.solve(T(L), npr.randn(*sample_shape))
   return np.linalg.solve(J, h)[...,None,:] + T(noise)

### packing and unpacking natural parameters and statistics into dense matrices

vs, hs = partial(np.concatenate, axis=-2), partial(np.concatenate, axis=-1)

def pack_dense(tup):
    '''Used for packing Gaussian natural parameters, Gaussian statistics, or NIW
    parameters into a dense ndarray so that we can use tensordot for all the
    linear contraction operations.'''
    # we don't use a symmetric embedding because factors of 1/2 on h are a pain
    J, h = tup[:2]
    leading_dim, N = h.shape[:-1], h.shape[-1]
    z1, z2 = np.zeros(leading_dim + (N, 1)), np.zeros(leading_dim + (1, 1))
    a, b = (z2, z2) if len(tup) == 2 else tup[2:]

    J = J[...,None] * np.eye(N)[None,...] if J.ndim == h.ndim else J
    h = h[...,None]
    a, b = np.reshape(a, leading_dim + (1, 1)), np.reshape(b, leading_dim + (1, 1))

    return vs(( hs(( J,     h,  z1 )),
                hs(( T(z1), a,  z2 )),
                hs(( T(z1), z2, b  ))))

def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return arr[...,:N, :N], arr[...,N,:N], arr[...,N,N], arr[...,N+1,N+1]
