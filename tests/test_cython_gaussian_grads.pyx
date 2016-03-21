# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

from svae.lds.cython_gaussian_grads cimport *
from svae.util import solve_triangular

# this file provides python wrappers for the cython functions for testing

F = lambda x: np.require(x, np.double, 'F')

def lognorm_grad_arg0(g, ans, L, v):
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    cdef double[::1] _v = np.require(v, np.double, 'F')

    cdef double[::1,:] out = np.zeros_like(L, order='F')
    _lognorm_grad_arg0(g, ans, _L, _v, out)
    return np.asarray(out)

def lognorm_grad_arg1(g, ans, L, v):
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    cdef double[::1] _v = np.require(v, np.double, 'F')

    cdef double[::1] out = np.zeros_like(v, order='F')
    _lognorm_grad_arg1(g, ans, _L, _v, out)
    return np.asarray(out)

def natural_predict(J, h, J11, J12, J22, logZ):
    cdef double[::1,:] _J = np.require(-2*J, np.double, 'F')
    cdef double[::1] _h = np.require(h, np.double)
    cdef double[::1,:] _J11 = np.require(-2*J11, np.double, 'F')
    cdef double[::1,:] _J12 = np.require(-J12, np.double, 'F')
    cdef double[::1,:] _J22 = np.require(-2*J22, np.double, 'F')
    cdef double _logZ = logZ

    cdef double[::1,:] J_predict = np.zeros_like(J, order='F')
    cdef double[::1] h_predict = np.zeros_like(h, order='F')
    cdef double lognorm

    cdef double [::1,:] L = np.zeros_like(J, order='F')
    cdef double[::1] v = np.zeros_like(h)
    cdef double[::1] v2 = np.zeros_like(h)
    cdef double[::1,:] temp = np.zeros_like(J, order='F')

    lognorm = _natural_predict(
        _J, _h, _J11, _J12, _J22, _logZ,
        J_predict, h_predict,
        L, v, v2, temp)

    return (-1./2 * np.asarray(J_predict), np.asarray(h_predict)), lognorm

def natural_predict_grad(g_J_predict, g_h_predict, g_lognorm, J12, L, v, v2, temp):
    cdef double[::1,:] _g_J_predict = np.require(-1./2*g_J_predict, np.double, 'F')
    cdef double[::1] _g_h_predict = np.require(g_h_predict, np.double)
    cdef double _g_lognorm = g_lognorm

    cdef double[::1,:] _J12 = np.require(J12, np.double, 'F')
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    cdef double[::1] _v = np.require(v, np.double)
    cdef double[::1] _v2 = np.require(v2, np.double)
    cdef double[::1,:] _temp = np.require(temp, np.double, 'F')

    cdef double[::1,:] g_J = np.zeros_like(g_J_predict, order='F')
    cdef double[::1] g_h = np.zeros_like(g_h_predict)

    cdef double[::1,:] temp_nn = np.zeros_like(L, order='F')
    cdef double[::1,:] temp_nn2 = np.zeros_like(L, order='F')
    cdef double[::1,:] temp_nn3 = np.zeros_like(L, order='F')
    cdef double[::1,:] g_temp = np.zeros_like(L, order='F')
    cdef double[::1] temp_n = np.zeros_like(v)
    cdef double[::1] temp_n2 = np.zeros_like(v)
    cdef double[::1] g_v = np.zeros_like(v)

    _natural_predict_grad(
        _g_J_predict, _g_h_predict, _g_lognorm,
        _J12, _L, _v, _v2, _temp,
        g_J, g_h,
        temp_nn, temp_nn2, temp_nn3, g_temp, temp_n, temp_n2, g_v)

    return -2*np.asarray(g_J), np.asarray(g_h)

def natural_lognorm_grad(g_lognorm, L, v):
    # inputs
    cdef double _g_lognorm = g_lognorm
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    cdef double[::1] _v = np.require(v, np.double, 'F')

    # outputs
    cdef double[::1,:] g_J = np.zeros_like(L, order='F')
    cdef double[::1] g_h = np.zeros_like(v, order='F')

    # temps
    cdef double[::1] g_v = np.zeros_like(v, order='F')
    cdef double[::1] temp_n = np.zeros_like(v, order='F')
    cdef double[::1,:] temp_nn = np.zeros_like(L, order='F')

    _natural_lognorm_grad(g_lognorm, g_J, g_h, _L, _v, g_v, temp_n, temp_nn)

    return np.asarray(g_J), np.asarray(g_h)

def rts_backward_step(next_smooth, next_pred, filtered, pair_param):
    _F = lambda x: map(F, x)

    # p = "predicted", f = "filtered", s = "smoothed", n = "next"
    (Jns, hns, mun), (Jnp, hnp), (Jf, hf) = _F(next_smooth), _F(next_pred), _F(filtered)
    J11, J12, J22, _ = _F(pair_param)

    # convert from natural parameter to the usual J definitions
    Jns, Jnp, Jf, J11, J12, J22 = -2*Jns, -2*Jnp, -2*Jf, -2*J11, -J12, -2*J22

    # outputs
    Js = np.zeros_like(Jns, order='F')
    hs = np.zeros_like(hns, order='F')
    Ex = np.zeros_like(hns, order='F')
    ExxT = np.zeros_like(Jns, order='F')
    ExnxT = np.zeros_like(Jns, order='F')

    # temps
    temp_nn = np.zeros_like(Jns, order='F')
    temp_nn2 = np.zeros_like(Jns, order='F')
    temp_n = np.zeros_like(hns, order='F')

    _rts_backward_step(
        Jns, hns, mun, Jnp, hnp, Jf, hf,
        J11, J12, J22,
        Js, hs, Ex, ExxT, ExnxT,
        temp_nn, temp_nn2, temp_n)

    stats = np.asarray(Ex), np.asarray(ExxT), np.asarray(ExnxT)
    return -1./2*np.asarray(Js), np.asarray(hs), stats

def rts_3_grad(*args):
    # map inputs to fortran order
    g_Ex, g_ExxT, g_ExnxT, Ex, ExxT, ExnxT, L, Sigma, mu, mun, J12 = map(F, args)

    # allocate new outputs
    g_L, g_mun = np.zeros_like(L, order='F'), np.zeros_like(mun, order='F')

    # allocate temps
    temp_nn, temp_nn2, temp_nn3, temp_nn4, temp_nn5, temp_nn6 = \
        [np.zeros_like(g_ExxT) for _ in range(6)]

    _rts_3_grad(
        g_Ex, g_ExxT, g_ExnxT,
        Ex, ExnxT,
        J12, mun,
        L,
        g_L, g_mun,
        temp_nn, temp_nn2, temp_nn3, temp_nn4, temp_nn5, temp_nn6)

    return np.asarray(g_L), np.asarray(g_ExxT), np.asarray(g_Ex), np.asarray(g_mun)

def rts_1_grad(*args):
    # map inputs to fortran order
    g_Js, g_hs, Js, hs, L, hns, hnp, Jf, hf, J11, J12 = map(F, args)

    # allocate new outputs
    g_L, g_Jf = [np.zeros_like(L, order='F') for _ in range(2)]
    g_hns, g_hnp, g_hf = [np.zeros_like(hns, order='F') for _ in range(3)]

    # compute intermediates
    temp = solve_triangular(L, np.asarray(J12).T)
    temp_n = solve_triangular(L, np.asarray(hns) - np.asarray(hnp))

    # allocate temps
    temp_nn, temp_nn2, temp_nn3, temp_nn4 = \
        [np.zeros_like(L, order='F') for _ in range(4)]

    _rts_1_grad(
        g_Js, g_hs,
        L, temp, temp_n,
        g_L, g_hns, g_hnp, g_Jf, g_hf,
        temp_nn, temp_nn2, temp_nn3, temp_nn4)

    return np.asarray(g_L), np.asarray(g_hns), np.asarray(g_hnp), np.asarray(g_Jf), np.asarray(g_hf)

def rts_backward_step_grad(
        # grads
        g_Js, g_hs, g_Ex, g_ExxT, g_ExnxT,
        # forward-pass args
        next_smooth, next_pred, filtered, pair_param,
        # forward-pass results
        Js, hs, stats):
    _F = lambda x: np.require(x, np.double, 'F')
    F = lambda x: map(_F, x)

    # convert incoming grads
    g_Js, g_hs, g_Ex, g_ExxT, g_ExnxT = F([g_Js, g_hs, g_Ex, g_ExxT, g_ExnxT])

    # p = "predicted", f = "filtered", s = "smoothed", n = "next"
    (Jns, hns, mun), (Jnp, hnp), (Jf, hf) = F(next_smooth), F(next_pred), F(filtered)
    J11, J12, J22, _ = F(pair_param)

    Js, hs, (Ex, ExxT, ExnxT) = _F(Js), _F(hs), F(stats)

    # compute intermediates
    L = _F(np.linalg.cholesky(Jns - Jnp + J22))
    temp = _F(solve_triangular(L, J12.T))
    temp_n = _F(solve_triangular(L, hns - hnp))

    # allocate outputs
    g_Jns = np.zeros_like(Jns)
    g_hns = np.zeros_like(hns)
    g_mun = np.zeros_like(mun)
    g_Jnp = np.zeros_like(Jnp)
    g_hnp = np.zeros_like(hnp)
    g_Jf = np.zeros_like(Jf)
    g_hf = np.zeros_like(hf)

    temp_nn = np.zeros_like(g_Js, order='F')
    temp_nn2 = np.zeros_like(g_Js, order='F')
    temp_nn3 = np.zeros_like(g_Js, order='F')
    temp_nn4 = np.zeros_like(g_Js, order='F')
    temp_nn5 = np.zeros_like(g_Js, order='F')
    temp_nn6 = np.zeros_like(g_Js, order='F')
    g_L = np.zeros_like(g_Js, order='F')

    _rts_backward_step_grad(
        # incoming grads
        g_Js, g_hs, g_Ex, g_ExxT, g_ExnxT,
        # forward-pass results
        Js, hs, Ex, ExxT, ExnxT,
        # forward-pass args subset
        J12, mun,
        # intermediates from forward pass
        L, temp, temp_n,
        # results
        g_Jns, g_hns, g_mun, g_Jnp, g_hnp, g_Jf, g_hf,
        # temps
        temp_nn, temp_nn2, temp_nn3, temp_nn4, temp_nn5, temp_nn6, g_L)

    g_next_smooth = np.asarray(g_Jns), np.asarray(g_hns), np.asarray(g_mun)
    g_next_pred = np.asarray(g_Jnp), np.asarray(g_hnp)
    g_filtered = np.asarray(g_Jf), np.asarray(g_hf)

    return g_next_smooth, g_next_pred, g_filtered

def info_to_mean_grad(g_mu, g_Sigma, J, h):
    g_mu = np.require(g_mu, np.double, 'F')
    g_Sigma = np.require(g_Sigma, np.double, 'F')
    J = np.require(J, np.double, 'F')
    h = np.require(h, np.double, 'F')

    mu = np.require(np.linalg.solve(J, h), np.double, 'F')
    Sigma = np.require(np.linalg.inv(J), np.double, 'F')

    temp_nn = np.zeros_like(J)
    temp_nn2 = np.zeros_like(J)

    g_J = np.zeros_like(J)
    g_h = np.zeros_like(h)

    _info_to_mean_grad(g_mu, g_Sigma, mu, Sigma, J, h, g_J, g_h, temp_nn, temp_nn2)

    return np.asarray(g_J), np.asarray(g_h)

def natural_sample(J, h, eps):
    cdef double[::1,:] _J = np.require(J, np.double, 'F')
    cdef double[::1,:] _h = np.require(h, np.double, 'F')
    cdef double[::1,:] _eps = np.require(eps, np.double, 'F')

    cdef double[::1,:] shaped_randvecs = np.zeros_like(h, order='F')

    cdef double[::1,:] L = np.zeros_like(J, order='F')
    cdef double[::1,:] temp_ns = np.zeros_like(h, order='F')

    _natural_sample(_J, _h, _eps, shaped_randvecs, L, temp_ns)

    return np.asarray(shaped_randvecs)

def natural_sample_grad(g, ans, J, h, eps):
    cdef double[::1,:] _g = np.require(g, np.double, 'F')
    cdef double[::1,:] _ans = np.require(ans, np.double, 'F')
    cdef double[::1,:] _J = np.require(J, np.double, 'F')
    cdef double[::1,:] _h = np.require(h, np.double, 'F')
    cdef double[::1,:] _eps = np.require(eps, np.double, 'F')

    cdef double[::1,:] L = np.require(np.linalg.cholesky(J), np.double, 'F')

    cdef double[::1,:] g_J = np.zeros_like(J, order='F')
    cdef double[::1,:] g_h = np.zeros_like(h, order='F')

    cdef double[::1,:] temp_nn  = np.zeros_like(J, order='F')
    cdef double[::1,:] temp_nn2 = np.zeros_like(J, order='F')
    cdef double[::1,:] temp_ns  = np.zeros_like(h, order='F')
    cdef double[::1,:] temp_ns2 = np.zeros_like(h, order='F')
    cdef double[::1,:] temp_ns3 = np.zeros_like(h, order='F')

    _natural_sample_grad(
        _g,
        _J, _h, L, _eps,
        g_J, g_h,
        temp_nn, temp_nn2, temp_ns, temp_ns2, temp_ns3)

    return np.asarray(g_J), np.asarray(g_h)
