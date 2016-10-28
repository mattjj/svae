# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython cimport floating
from scipy.linalg.cython_lapack cimport dtrtrs, strtrs, dpotri, spotri

def solve_triangular(floating[:,:,::1] L, floating[:,:,::1] X, trans, lower):
    cdef int i, K = L.shape[0]
    cdef floating[:,:,::1] out = np.copy(np.swapaxes(X, -1, -2), 'C')

    # flip these because we're working in C order, while LAPACK is F order
    cdef char _trans = 'T' if trans not in (1, 'T') else 'N'
    cdef char _lower = 'U' if lower else 'L'

    for i in range(K):
        _solve_triangular(L[i,:,:], X[i,:,:], out[i,:,:], _trans, _lower)

    return np.swapaxes(out, -1, -2)

cdef inline void _solve_triangular(
        floating[:,::1] L, floating[:,::1] X, floating[:,::1] out,
        char trans, char lower) nogil:
    cdef int M = X.shape[0], N = X.shape[1], info

    if floating is double:
        dtrtrs(&lower, &trans, 'N', &M, &N, &L[0,0], &M, &out[0,0], &M, &info)
    else:
        strtrs(&lower, &trans, 'N', &M, &N, &L[0,0], &M, &out[0,0], &M, &info)


def inv_posdef_from_cholesky(floating[:,:,::1] L, lower):
    cdef int i, K = L.shape[0]
    cdef floating[:,:,::1] out = np.copy(L, 'C')

    # flip this because we're working in C order, while LAPACK is F order
    cdef char _lower = 'U' if lower else 'L'

    for i in range(K):
        _inv_posdef_from_cholesky(out[i,:,:], _lower)

    return np.asarray(out)

cdef inline void _inv_posdef_from_cholesky(floating[:,::1] L, char lower):
    cdef int N = L.shape[0], info

    if floating is double:
        dpotri(&lower, &N, &L[0,0], &N, &info)
    else:
        spotri(&lower, &N, &L[0,0], &N, &info)

    copy_lower_upper(L)


cdef inline void copy_lower_upper(floating[:,::1] L):
    cdef int i, j, N = L.shape[0]
    for i in range(N):
        for j in range(i):
            L[j,i] = L[i,j]
