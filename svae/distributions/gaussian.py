from __future__ import division
import autograd.numpy as np

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
        - np.sum(a) - np.sum(b)

### packing and unpacking node potentials into dense matrices

vs, hs = partial(np.concatenate, axis=-2), partial(np.concatenate, axis=-1)
tr = partial(np.swapaxes, axis1=-1, axis2=-2)

def pack_dense(node_potentials):
    '''Packs (J, h, a, b) tuple, where J has shape (T, N), h has shape (T, N),
    and a and b both have shape (T, 1), into a dense ndarray of shape
    (T, N+2, N+2), so we can use tensordot. If argument is of the form (J, h),
    then a and b are set to zero.'''
    if len(node_potentials) not in {2, 4}: raise ValueError

    Jdiag, h = node_potentials[:2]
    J = Jdiag[...,None] * np.eye(N)[None,...]
    h = 1./2 * h[...,None]
    T, N = h.shape
    z1, z2 = np.zeros((T, N, 1)), np.zeros((T, 1, 1))
    a, b = node_potentials[2:] if len(node_potentials) == 4 else (z2, z2)

    return vs(( hs(( J,      h,  z1 )),
                hs(( tr(h),  a,  z2 )),
                hs(( tr(z1), z2, b  ))))

def unpack_dense(arr):
    '''Unpacks dense ndarray of shape (T, N+2, N+2) into four parts, with shapes
      (T, N, N), (T, N), (T,), and (T,).'''
    T, N = arr.shape[0], arr.shape[1] - 2
    return arr[:,:N, :N], arr[:,N,:N], arr[:,N,N], arr[:,N+1,N+1]
