# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import numpy.random as npr
from scipy.linalg.cython_blas cimport ddot, dgemm, dgemv

from libc.math cimport exp, log

from svae.cython_util cimport *


### util

cdef inline double normalize_inplace(double[::1] a):
    cdef int i, N = a.shape[0]
    cdef double tot = 0.

    for i in range(N):
        tot += a[i]
    for i in range(N):
        a[i] /= tot

    return tot


cdef inline double logsumexp(double[::1] a):
    cdef int i, N = a.shape[0]
    cdef double tot, cmax

    cmax = a[0]
    for i in range(1, N):
        cmax = max(cmax, a[i])

    tot = 0.
    for i in range(N):
        tot += exp(a[i] - cmax)

    return log(tot) + cmax


cdef inline void logsumexp_grad(double g, double ans, double[::1] a, double[::1] g_a):
    cdef int i, N = a.shape[0]

    for i in range(N):
        g_a[i] += g * exp(a[i] - ans)


cdef inline void add_vectors(double[::1] a, double[::1] b, double[::1] out):
    cdef int i, N = a.shape[0]

    for i in range(N):
        out[i] = a[i] + b[i]


### normalized forward messages

def hmm_logZ_normalized(natparam):
    # inputs
    _init_params, _pair_params, _node_params = natparam
    cdef double[::1] init_params = np.exp(np.require(_init_params, np.double))
    cdef double[::1,:] pair_params = np.exp(np.require(_pair_params, np.double, 'F'))
    cdef double[:,::1] node_params = np.require(_node_params, np.double, 'C')
    cdef int T = _node_params.shape[0], N = _node_params.shape[1]

    # outputs
    cdef double lognorm = 0.

    # intermediates (TODO keep for backward pass? if not, don't need to save all of these)
    cdef double[:,::1] alpha = np.zeros((T, N), order='C')

    # temps
    cdef double[::1] in_potential = np.zeros(N)
    cdef double themax
    cdef int i, t, inc = 1
    cdef double one = 1., zero = 0.

    copy_vector(init_params, in_potential)
    for t in range(T):
        themax = max_vector(node_params[t])
        for i in range(N):
            alpha[t,i] = in_potential[i] * exp(node_params[t,i] - themax)
        lognorm += log(normalize_inplace(alpha[t])) + themax
        dgemv('T', &N, &N, &one, &pair_params[0,0], &N, &alpha[t,0], &inc,
              &zero, &in_potential[0], &inc)

    return lognorm, alpha


### log forward messages

def hmm_logZ(natparam):
    _init_params, _pair_params, _node_params = natparam
    cdef double[::1] init_params = np.require(_init_params, np.double)
    cdef double[::1,:] pair_params = np.require(_pair_params, np.double, 'F')
    cdef double[:,::1] node_params = np.require(_node_params, np.double, 'C')
    cdef int T = _node_params.shape[0], N = _node_params.shape[1]

    # outputs
    cdef double logZ

    # intermediates
    cdef double[:,::1] log_alpha = np.zeros((T, N), order='C')

    # temps
    cdef int i, t
    cdef double[::1] temp = np.zeros(N)

    for i in range(N):
        log_alpha[0,i] = init_params[i] + node_params[0,i]

    for t in range(1, T):
        for i in range(N):
            add_vectors(log_alpha[t-1], pair_params[:,i], temp)
            log_alpha[t,i] += logsumexp(temp) + node_params[t,i]

    logZ = logsumexp(log_alpha[T-1])

    intermediates = logZ, log_alpha, pair_params
    return logZ, intermediates


### gradient using intermediates from log forward messages

def hmm_logZ_grad(_g, intermediates):
    # input
    cdef double g = _g

    # intermediates
    _logZ, _log_alpha, _pair_params = intermediates
    cdef double logZ = _logZ
    cdef double[:,::1] log_alpha = _log_alpha
    cdef double[::1,:] pair_params = _pair_params
    cdef int T = _log_alpha.shape[0], N = _log_alpha.shape[1]

    # outputs
    cdef double[::1] g_init_params = np.zeros(N)
    cdef double[::1,:] g_pair_params = np.zeros((N, N), order='F')
    cdef double[:,::1] g_node_params = np.zeros((T, N), order='C')

    # temps
    cdef int i, t
    cdef double[:,::1] g_alpha = np.zeros((T, N), order='C')
    cdef double ans
    cdef double[::1] temp = np.zeros(N)
    cdef double[::1] g_temp = np.zeros(N)

    logsumexp_grad(g, logZ, log_alpha[T-1], g_alpha[T-1])
    for t in range(T-1, 0, -1):
        add_into_vector(g_alpha[t], g_node_params[t])
        for i in range(N):
            # reconstruct temp and ans from forward pass
            add_vectors(log_alpha[t-1], pair_params[:,i], temp)
            ans = logsumexp(temp)

            # apply logsumexp grad
            zero_vector(g_temp)
            logsumexp_grad(g_alpha[t,i], ans, temp, g_temp)
            add_into_vector(g_temp, g_alpha[t-1])
            add_into_vector(g_temp, g_pair_params[:,i])

    add_into_vector(g_alpha[0], g_init_params)
    add_into_vector(g_alpha[0], g_node_params[0])

    return np.asarray(g_init_params), np.asarray(g_pair_params), np.asarray(g_node_params)
