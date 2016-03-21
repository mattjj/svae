# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import numpy.random as npr
from scipy.linalg.cython_blas cimport dsymv, ddot, dgemm
from scipy.linalg.cython_lapack cimport dtrtrs, dpotrf, dpotrs, dpotri
cimport cython

from svae.cython_util cimport *
from svae.cython_linalg_grads cimport *
from cython_gaussian_grads cimport *

### filtering

def ensure_time_axis(pair_params):
    # TODO could also ensure dense here (list of ndarrays and floats, not list of tuples)
    J11, J12, J22, logZ = pair_params
    homog = J11.ndim == 2

    if homog:
        return map(lambda x: np.expand_dims(x, -1), pair_params), 0

    T = lambda x: np.transpose(x, (1, 2, 0))
    return (T(J11), T(J12), T(J22), logZ), 1

def natural_filter_forward_general(init_params, pair_params, node_params):
    # inputs
    cdef double[::1,:] Jinit = np.require(-2*init_params[0], np.double, 'F')
    cdef double[::1] hinit = np.require(init_params[1], np.double, 'F')
    cdef double logZ_init = init_params[2]

    pair_params, _step = ensure_time_axis(pair_params)
    _J11, _J12, _J22, _logZ_pair = pair_params
    cdef double[::1,:,:] J11 = np.require(-2*_J11, np.double, 'F')
    cdef double[::1,:,:] J12 = np.require(-_J12, np.double, 'F')
    cdef double[::1,:,:] J22 = np.require(-2*_J22, np.double, 'F')
    cdef double[::1] logZ_pair = np.require(_logZ_pair, np.double, 'F')
    cdef int step = _step

    _J_node, _h_node, _logZ_node = node_params
    cdef double[::1,:] J_node = -2*np.require(np.asarray(_J_node).T, np.double, 'F')
    cdef double[::1,:] h_node = np.require(np.asarray(_h_node).T, np.double, 'F')
    cdef double[::1] logZ_node = np.require(_logZ_node, np.double)

    cdef int n = Jinit.shape[0], T = J_node.shape[1], t

    # outputs
    cdef double[::1,:,:] J_predict = np.zeros((n, n, T), order='F')
    cdef double[::1,:] h_predict = np.zeros((n, T), order='F')
    cdef double[::1,:,:] J_filtered = np.zeros((n, n, T), order='F')
    cdef double[::1,:] h_filtered = np.zeros((n, T), order='F')
    cdef double lognorm = 0.

    # 'temps' we output for the reverse pass to use
    cdef double[::1,:,:] L = np.zeros((n, n, T), order='F')
    cdef double[::1,:] v = np.zeros((n, T), order='F')
    cdef double[::1,:] v2 = np.zeros((n, T-1), order='F')
    cdef double[::1,:,:] temp = np.zeros((n, n, T-1), order='F')

    copy(Jinit, J_predict[:,:,0])
    copy_vector(hinit, h_predict[:,0])
    lognorm = logZ_init

    for t in range(T-1):
        lognorm += _natural_condition_diag(
                J_predict[:,:,t], h_predict[:,t],
                J_node[:,t], h_node[:,t], logZ_node[t],
                J_filtered[:,:,t], h_filtered[:,t])
        lognorm += _natural_predict(
            J_filtered[:,:,t], h_filtered[:,t],
            J11[:,:,t*step], J12[:,:,t*step], J22[:,:,t*step], logZ_pair[t*step],
            J_predict[:,:,t+1], h_predict[:,t+1],
            L[:,:,t], v[:,t], v2[:,t], temp[:,:,t])

    lognorm += _natural_condition_diag(
        J_predict[:,:,T-1], h_predict[:,T-1],
        J_node[:,T-1], h_node[:,T-1], logZ_node[T-1],
        J_filtered[:,:,T-1], h_filtered[:,T-1])

    lognorm += _natural_lognorm(J_filtered[:,:,T-1], h_filtered[:,T-1], L[:,:,T-1], v[:,T-1])

    prediction_messages = -1./2*np.asarray(J_predict).T, np.asarray(h_predict).T
    filtered_messages = -1./2*np.asarray(J_filtered).T, np.asarray(h_filtered).T

    result = (prediction_messages, filtered_messages), lognorm

    intermediates = J12, L, v, v2, temp, step
    return result, intermediates

def natural_filter_grad(g, intermediates):
    # input
    ((_g_J_predict, _g_h_predict), (_g_J_filtered, _g_h_filtered)), _g_lognorm = g
    cdef double[::1,:,:] g_J_predict = np.require(-1./2*_g_J_predict.T, np.double, 'F')
    cdef double[::1,:] g_h_predict = np.require(_g_h_predict.T, np.double, 'F')
    cdef double[::1,:,:] g_J_filtered = np.require(-1./2*_g_J_filtered.T, np.double, 'F')
    cdef double[::1,:] g_h_filtered = np.require(_g_h_filtered.T, np.double, 'F')
    cdef double g_lognorm = _g_lognorm

    J12, L, v, v2, temp, step = intermediates
    cdef double[::1,:,:] _J12 = J12
    cdef double[::1,:,:] _L = L
    cdef double[::1,:] _v = v
    cdef double[::1,:] _v2 = v2
    cdef double[::1,:,:] _temp = temp
    cdef int _step = step

    cdef int T = _L.shape[2], n = L.shape[0], t

    # output
    cdef double[::1,:] g_J_node = np.zeros((n, T), order='F')
    cdef double[::1,:] g_h_node = np.zeros((n, T), order='F')
    cdef double[::1] g_logZ_node = np.repeat(_g_lognorm, T)

    # temps
    cdef double[::1,:] temp_nn = np.zeros((n,n), order='F')
    cdef double[::1,:] temp_nn2 = np.zeros((n,n), order='F')
    cdef double[::1,:] temp_nn3 = np.zeros((n,n), order='F')
    cdef double[::1,:] g_temp = np.zeros((n,n), order='F')
    cdef double[::1] temp_n = np.zeros(n)
    cdef double[::1] temp_n2 = np.zeros(n)
    cdef double[::1] g_v = np.zeros(n)

    _natural_lognorm_grad(
        g_lognorm,                                     # input
        g_J_filtered[:,:,T-1], g_h_filtered[:,T-1],    # output
        _L[:,:,T-1], _v[:,T-1], g_v, temp_n, temp_nn)  # side info and temps
    _natural_condition_diag_grad(
        g_J_filtered[:,:,T-1], g_h_filtered[:,T-1],  # input
        g_J_predict[:,:,T-1], g_h_predict[:,T-1],    # output
        g_J_node[:,T-1], g_h_node[:,T-1])            # output
    for t in range(T-1, 0, -1):
        _natural_predict_grad(
            g_J_predict[:,:,t], g_h_predict[:,t], g_lognorm,            # input
            _J12[:,:,(t-1)*_step],
            _L[:,:,t-1], _v[:,t-1], _v2[:,t-1], _temp[:,:,t-1],         # side info
            g_J_filtered[:,:,t-1], g_h_filtered[:,t-1],                 # output
            temp_nn, temp_nn2, temp_nn3, g_temp, temp_n, temp_n2, g_v)  # temps
        _natural_condition_diag_grad(
            g_J_filtered[:,:,t-1], g_h_filtered[:,t-1],  # input
            g_J_predict[:,:,t-1], g_h_predict[:,t-1],    # output
            g_J_node[:,t-1], g_h_node[:,t-1])            # output

    return -2*np.asarray(g_J_node).T, np.asarray(g_h_node).T, np.asarray(g_logZ_node)

### smoothing

def natural_smoother_general(forward_messages, pair_params):
    prediction_messages, filtered_messages = forward_messages
    _J_predict, _h_predict = prediction_messages
    _J_filtered, _h_filtered = filtered_messages
    cdef double[::1,:,:] J_predict = -2*_J_predict.T
    cdef double[::1,:] h_predict = _h_predict.T
    cdef double[::1,:,:] J_filtered = -2*_J_filtered.T
    cdef double[::1,:] h_filtered = _h_filtered.T

    pair_params, _step = ensure_time_axis(pair_params)
    _J11, _J12, _J22, _ = pair_params
    cdef double[::1,:,:] J11 = np.require(-2*_J11, np.double, 'F')
    cdef double[::1,:,:] J12 = np.require(-_J12, np.double, 'F')
    cdef double[::1,:,:] J22 = np.require(-2*_J22, np.double, 'F')
    cdef int step = _step

    cdef int T = J_predict.shape[2], n = J_predict.shape[0]
    cdef double[::1,:,:] Js = np.zeros((n, n, T), order='F')
    cdef double[::1,:] hs = np.zeros((n, T), order='F')

    cdef double[::1,:] Ex = np.zeros((n, T), order='F')
    cdef double[::1,:,:] ExxT = np.zeros((n, n, T), order='F')
    cdef double[::1,:,:] ExnxT = np.zeros((n, n, T-1), order='F')

    cdef double[::1,:,:] Ls = np.zeros((n, n, T-1), order='F')
    cdef double[::1,:,:] temp_nns = np.zeros((n, n, T-1), order='F')
    cdef double[::1,:] temp_ns = np.zeros((n, T-1), order='F')
    cdef int t, i, j

    copy(J_filtered[:,:,T-1], Js[:,:,T-1])
    copy_vector(h_filtered[:,T-1], hs[:,T-1])
    _info_to_mean(J_filtered[:,:,T-1], h_filtered[:,T-1], Ex[:,T-1], ExxT[:,:,T-1])
    for j in range(n):
        for i in range(n):
            ExxT[i,j,T-1] += Ex[i,T-1] * Ex[j,T-1]

    for t in range(T-1, 0, -1):
        _rts_backward_step(
            Js[:,:,t], hs[:,t], Ex[:,t], J_predict[:,:,t], h_predict[:,t],
            J_filtered[:,:,t-1], h_filtered[:,t-1],
            J11[:,:,(t-1)*step], J12[:,:,(t-1)*step], J22[:,:,(t-1)*step],
            Js[:,:,t-1], hs[:,t-1], Ex[:,t-1], ExxT[:,:,t-1], ExnxT[:,:,t-1],
            Ls[:,:,t-1], temp_nns[:,:,t-1], temp_ns[:,t-1])

    ans = Js, hs, (Ex, ExxT, ExnxT)
    intermediates = Ls, temp_nns, temp_ns, J12, step, ans
    return _compute_stats(Ex, ExxT, ExnxT, step), intermediates

@cython.wraparound(True)
def _compute_stats(
        double[::1,:] Ex, double[::1,:,:] ExxT, double[::1,:,:] ExnxT, int inhomog):
    cdef int T = Ex.shape[1]
    E_init_stats = np.asarray(ExxT[:,:,0]), np.asarray(Ex[:,0]), 1., 1.
    E_pair_stats = np.transpose(ExxT, (2, 0, 1))[:-1], \
        np.asarray(ExnxT).T, np.transpose(ExxT, (2, 0, 1))[1:], np.ones(T-1)
    E_node_stats = np.diagonal(np.asarray(ExxT).T, axis1=-1, axis2=-2), \
        np.asarray(Ex).T, np.ones(T)

    if not inhomog:
        E_pair_stats = map(lambda x: np.sum(x, axis=0), E_pair_stats)

    return E_init_stats, E_pair_stats, E_node_stats

@cython.wraparound(True)
def _compute_stats_grad(g):
    g_E_init_stats, g_E_pair_stats, g_E_node_stats = g

    # node stats
    g_ExxT, g_Ex, _ = g_E_node_stats
    T, N = g_Ex.shape[0], g_Ex.shape[1]
    g_ExxT = g_ExxT[:,:,None] * np.eye(N)[None,:,:]

    # init stats
    g_Ex1x1T, g_Ex1 = g_E_init_stats[:2]
    g_Ex[0] += g_Ex1
    g_ExxT[0] += g_Ex1x1T

    # pair stats
    g_ExpxpT, g_ExxnT, g_ExnxnT, _ = g_E_pair_stats
    g_ExxT[:-1] += g_ExpxpT
    g_ExxT[1:] += g_ExnxnT
    homog = g_ExxnT.ndim == 2
    if homog:
        g_ExxnT = np.repeat(g_ExxnT[None,:,:], T-1, axis=0)

    return g_Ex.T, np.transpose(g_ExxT, (1, 2, 0)), g_ExxnT.T

def natural_smoother_general_grad(g, intermediates):
    # grads
    _g_Ex, _g_ExxT, _g_ExnxT = _compute_stats_grad(g)
    cdef double[::1,:] g_Ex = np.require(_g_Ex, np.double, 'F')
    cdef double[::1,:,:] g_ExxT = np.require(_g_ExxT, np.double, 'F')
    cdef double[::1,:,:] g_ExnxT = np.require(_g_ExnxT, np.double, 'F')

    cdef int N = g_ExxT.shape[0], T = g_ExxT.shape[2]

    # intermediates (including ans, instead of using the one passed as argument)
    _L, _temp_nn, _temp_n, _J12, _step, ans = intermediates
    cdef double[::1,:,:] L = _L
    cdef double[::1,:,:] temp_nn = _temp_nn
    cdef double[::1,:] temp_n = _temp_n
    cdef double[::1,:,:] J12 = _J12
    cdef int step = _step

    _Js, _hs, (_Ex, _ExxT, _ExnxT) = ans
    cdef double[::1,:,:] Js = _Js
    cdef double[::1,:] hs = _hs
    cdef double[::1,:] Ex = _Ex
    cdef double[::1,:,:] ExxT = _ExxT
    cdef double[::1,:,:] ExnxT = _ExnxT

    # outputs
    cdef double[::1,:,:] g_J_predict = np.zeros((N, N, T), order='F')
    cdef double[::1,:] g_h_predict = np.zeros((N, T), order='F')
    cdef double[::1,:,:] g_J_filtered = np.zeros((N, N, T), order='F')
    cdef double[::1,:] g_h_filtered = np.zeros((N, T), order='F')

    # temps
    cdef double[::1,:] temp_nn1 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn2 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn3 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn4 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn5 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn6 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn7 = np.zeros((N, N), order='F')
    cdef double[::1,:,:] g_Js = np.zeros((N, N, T), order='F')
    cdef double[::1,:] g_hs = np.zeros((N, T), order='F')
    cdef int t, i, j

    for t in range(T-1):
        _rts_backward_step_grad(
            g_Js[:,:,t], g_hs[:,t], g_Ex[:,t], g_ExxT[:,:,t], g_ExnxT[:,:,t],
            Js[:,:,t], hs[:,t], Ex[:,t], ExxT[:,:,t], ExnxT[:,:,t],
            J12[:,:,t*step], Ex[:,t+1],
            L[:,:,t], temp_nn[:,:,t], temp_n[:,t],
            g_Js[:,:,t+1], g_hs[:,t+1], g_Ex[:,t+1],
            g_J_predict[:,:,t+1], g_h_predict[:,t+1],
            g_J_filtered[:,:,t], g_h_filtered[:,t],
            temp_nn1, temp_nn2, temp_nn3, temp_nn4, temp_nn5, temp_nn6, temp_nn7,
            )
    for j in range(N):
        for i in range(N):
            g_Ex[i,T-1] += g_ExxT[i,j,T-1] * Ex[j,T-1]
            g_Ex[j,T-1] += g_ExxT[i,j,T-1] * Ex[i,T-1]
    copy(ExxT[:,:,T-1], temp_nn2)
    for j in range(N):
        for i in range(N):
            temp_nn2[i,j] -= Ex[i,T-1] * Ex[j,T-1]
    _info_to_mean_grad(
        g_Ex[:,T-1], g_ExxT[:,:,T-1], Ex[:,T-1], temp_nn2, Js[:,:,T-1], hs[:,T-1],
        g_J_filtered[:,:,T-1], g_h_filtered[:,T-1], temp_nn1, temp_nn3)
    add_into_vector(g_hs[:,T-1], g_h_filtered[:,T-1])
    add_into(g_Js[:,:,T-1], g_J_filtered[:,:,T-1])

    g_prediction_messages = -2*np.transpose(g_J_predict, (2, 0, 1)), np.asarray(g_h_predict).T
    g_filtered_messages = -2*np.transpose(g_J_filtered, (2, 0, 1)), np.asarray(g_h_filtered).T

    return g_prediction_messages, g_filtered_messages

### sampling

def natural_sample_backward(forward_messages, pair_params, num_samples):
    # inputs
    _, filtered_messages = forward_messages
    _J_filtered, _h_filtered = filtered_messages
    cdef double[::1,:,:] J_filtered = -2*_J_filtered.T
    cdef double[::1,:] h_filtered = _h_filtered.T

    pair_params, _step = ensure_time_axis(pair_params)
    _J11, _J12, _, _, = pair_params
    cdef double[::1,:,:] J11 = np.require(-2*_J11, np.double, 'F')
    cdef double[::1,:,:] J12 = np.require(-_J12, np.double, 'F')
    cdef int step = _step

    cdef int T = J_filtered.shape[2], N = J_filtered.shape[0], S = num_samples
    cdef int i, j, t

    # outputs
    cdef double[::1,:,:] samples = np.zeros((N, S, T), order='F')

    # intermediates reused in backward pass
    cdef double[::1,:,:] J = np.zeros((N, N, T), order='F')
    cdef double[::1,:,:] h = np.zeros((N, S, T), order='F')
    cdef double[::1,:,:] L = np.zeros((N, N, T), order='F')
    cdef double[::1,:,:] eps = np.require(np.flipud(npr.randn(T, S, N)).T, np.double, 'F')

    # temps
    cdef double[::1,:] temp_ns = np.zeros((N, S), order='F')

    copy(J_filtered[:,:,T-1], J[:,:,T-1])
    for j in range(S):
        for i in range(N):
            h[i,j,T-1] = h_filtered[i,T-1]
    _natural_sample(
        J[:,:,T-1], h[:,:,T-1], eps[:,:,T-1], samples[:,:,T-1],
        L[:,:,T-1], temp_ns)
    for t in range(T-1, 0, -1):
        _natural_condition_on(
            J_filtered[:,:,t-1], h_filtered[:,t-1],
            samples[:,:,t], J11[:,:,(t-1)*step], J12[:,:,(t-1)*step],
            J[:,:,t-1], h[:,:,t-1])
        _natural_sample(
            J[:,:,t-1], h[:,:,t-1], eps[:,:,t-1], samples[:,:,t-1],
            L[:,:,t-1], temp_ns)

    intermediates = J, h, L, eps, J12, step
    return np.asarray(samples).T, intermediates

def natural_sample_backward_grad(_g_samples, intermediates):
    cdef double[::1,:,:] g_samples = _g_samples.T

    _J, _h, _L, _eps, _J12, _step = intermediates
    cdef double[::1,:,:] J = _J
    cdef double[::1,:,:] h = _h
    cdef double[::1,:,:] L = _L
    cdef double[::1,:,:] eps = _eps
    cdef double[::1,:,:] J12 = _J12
    cdef int step = _step

    cdef int T = J.shape[2], N = J.shape[0], S = g_samples.shape[1]

    # output
    cdef double[::1,:,:] g_J_predict = np.zeros((N, N, T), order='F')
    cdef double[::1,:] g_h_predict = np.zeros((N, T), order='F')
    cdef double[::1,:,:] g_J_filtered = np.zeros((N, N, T), order='F')
    cdef double[::1,:] g_h_filtered = np.zeros((N, T), order='F')

    # temps
    cdef double[::1,:,:] g_J = np.zeros_like(_J, order='F')
    cdef double[::1,:,:] g_h = np.zeros_like(_h, order='F')
    cdef double[::1,:] temp_nn  = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_nn2 = np.zeros((N, N), order='F')
    cdef double[::1,:] temp_ns  = np.zeros((N, S), order='F')
    cdef double[::1,:] temp_ns2 = np.zeros((N, S), order='F')
    cdef double[::1,:] temp_ns3 = np.zeros((N, S), order='F')
    cdef int t, i, j

    for t in range(T-1):
        _natural_sample_grad(
            g_samples[:,:,t],
            J[:,:,t], h[:,:,t], L[:,:,t], eps[:,:,t],
            g_J[:,:,t], g_h[:,:,t],
            temp_nn, temp_nn2, temp_ns, temp_ns2, temp_ns3)
        _natural_condition_on_grad(
            g_J[:,:,t], g_h[:,:,t],
            g_J_filtered[:,:,t], g_h_filtered[:,t], g_samples[:,:,t+1],
            J12[:,:,t*step])
    _natural_sample_grad(
        g_samples[:,:,T-1],
        J[:,:,T-1], h[:,:,T-1], L[:,:,T-1], eps[:,:,T-1],
        g_J[:,:,T-1], g_h[:,:,T-1],
        temp_nn, temp_nn2, temp_ns, temp_ns2, temp_ns3)
    for j in range(S):
        for i in range(N):
            g_h_filtered[i,T-1] += g_h[i,j,T-1]
    add_into(g_J[:,:,T-1], g_J_filtered[:,:,T-1])

    g_predict_messages = -2*np.asarray(g_J_predict).T, np.asarray(g_h_predict).T
    g_filtered_messages = -2*np.asarray(g_J_filtered).T, np.asarray(g_h_filtered).T

    return g_predict_messages, g_filtered_messages
