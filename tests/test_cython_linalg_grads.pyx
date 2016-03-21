# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython cimport floating
import scipy.linalg

from svae.cython_linalg_grads cimport *


al2d = lambda x: x if x.ndim > 1 else x[:,None]

def solve_triangular(L, v, trans='N'):
    return scipy.linalg.solve_triangular(L, v, lower=True, trans=trans)

def solve_triangular_grad_arg0(g, ans, a, b, trans):
    cdef double[::1,:] _g = np.require(al2d(g), np.double, 'F')
    cdef double[::1,:] _ans = np.require(al2d(ans), np.double, 'F')
    cdef double[::1,:] _a = np.require(a, np.double, 'F')
    cdef double[::1,:] _b = np.require(al2d(b), np.double, 'F')

    cdef double[::1,:] out = np.zeros_like(a, order='F')
    cdef double[::1,:] temp = np.zeros_like(al2d(g), order='F')
    cdef double[::1,:] temp2 = np.zeros_like(a, order='F')
    _solve_triangular_grad_arg0(_g, _ans, _a, _b, trans == 'T', out, temp, temp2)
    return np.squeeze(np.asarray(out))

def solve_triangular_grad_arg1(g, ans, a, b, trans):
    cdef double[::1,:] _g = np.require(al2d(g), np.double, 'F')
    cdef double[::1,:] _ans = np.require(al2d(ans), np.double, 'F')
    cdef double[::1,:] _a = np.require(a, np.double, 'F')
    cdef double[::1,:] _b = np.require(al2d(b), np.double, 'F')

    cdef double[::1,:] out = np.zeros_like(al2d(b), order='F')
    cdef double[::1,:] temp = np.zeros_like(al2d(b), order='F')
    _solve_triangular_grad_arg1(_g, _ans, _a, _b, trans == 'T', out, temp)
    return np.squeeze(np.asarray(out))

def cholesky_grad(floating[:,:] L, floating[:,:] dL):
    cdef double[::1,:] _dL = np.require(np.tril(dL), np.double, 'F')
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    _cholesky_grad(_L, _dL)
    return np.asarray(_dL)

def dpotrs_grad(g, ans, L, h):
    cdef double[::1,:] _g = np.require(g, np.double, 'F')
    cdef double[::1,:] _ans = np.require(ans, np.double, 'F')
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    cdef double[::1,:] _h = np.require(h, np.double, 'F')

    cdef double[::1,:] inter_ans = np.require(solve_triangular(L, h), np.double, 'F')

    cdef double[::1,:] g_J = np.zeros_like(L, order='F')
    cdef double[::1,:] g_h = np.zeros_like(h, order='F')

    cdef double[::1,:] temp_nn  = np.zeros_like(L, order='F')
    cdef double[::1,:] temp_ns  = np.zeros_like(h, order='F')
    cdef double[::1,:] temp_ns2 = np.zeros_like(h, order='F')

    _dpotrs_grad(_g, _ans, _L, inter_ans, g_J, g_h, temp_nn, temp_ns, temp_ns2)

    return np.asarray(g_J), np.asarray(g_h)
