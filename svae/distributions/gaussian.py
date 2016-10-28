from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from functools import partial

from svae.util import T, cholesky, outer, solve_tri, \
    solve_posdef_from_cholesky, inv_posdef_from_cholesky

def expectedstats(natparam, L=None):
    neghalfJ, h, _, _ = unpack_dense(natparam)
    L = cholesky(-2*neghalfJ) if L is None else L
    Ex = solve_posdef_from_cholesky(L, h)
    ExxT = inv_posdef_from_cholesky(L) + outer(Ex, Ex)
    En = np.ones(L.shape[:-2])
    return pack_dense(ExxT, Ex, En, En)

def logZ(natparam, L=None):
    neghalfJ, h, a, b = unpack_dense(natparam)
    L = cholesky(-2*neghalfJ) if L is None else L
    Linv_h = np.ravel(solve_tri(L, h))
    return 1./2 * np.dot(Linv_h, Linv_h) \
        - np.sum(np.log(np.diagonal(L, axis1=-1, axis2=-2))) \
        + np.sum(a + b)

def inference(natparam):
    L = cholesky(-2*neghalfJ) if L is None else L
    return expectedstats(natparam, L), logZ(natparam, L)

def natural_sample(natparam, num_samples):
    neghalfJ, h, _, _ = unpack_dense(natparam)
    sample_shape = np.shape(h) + (num_samples,)
    L = cholesky(-2*neghalfJ)
    Ex = solve_posdef_from_cholesky(L, h)
    noise = solve_tri(L, npr.randn(*sample_shape), trans='T')
    return Ex[...,None,:] + T(noise)

### packing and unpacking natural parameters and statistics into dense matrices

vs, hs = partial(np.concatenate, axis=-2), partial(np.concatenate, axis=-1)

def pack_dense(A, b, *args):
    '''Used for packing Gaussian natural parameters and statistics into a dense
    ndarray so that we can use tensordot for all the linear contraction ops.'''
    # we don't use a symmetric embedding because factors of 1/2 on h are a pain
    leading_dim, N = b.shape[:-1], b.shape[-1]
    z1, z2 = np.zeros(leading_dim + (N, 1)), np.zeros(leading_dim + (1, 1))
    c, d = args if args else (z2, z2)

    A = A[...,None] * np.eye(N)[None,...] if A.ndim == b.ndim else A
    b = b[...,None]
    c, d = np.reshape(c, leading_dim + (1, 1)), np.reshape(d, leading_dim + (1, 1))

    return vs(( hs(( A,     b,  z1 )),
                hs(( T(z1), c,  z2 )),
                hs(( T(z1), z2, d  ))))

def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return arr[...,:N, :N], arr[...,:N,N], arr[...,N,N], arr[...,N+1,N+1]
