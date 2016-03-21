# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

from scipy.linalg.cython_blas cimport dsymv, ddot, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dtrtrs, dpotrf, dpotrs, dpotri

from cython_util cimport *

### cholesky

cdef inline void _cholesky_grad(double[::1,:] L, double[::1,:] dL) nogil:
    cdef int i, k, n, N = L.shape[0], inc = 1
    cdef double neg1 = -1, one = 1

    dL[N-1,N-1] /= 2. * L[N-1,N-1]
    for k in range(N-2, -1, -1):
        n = N-k-1
        dsymv('L', &n, &neg1, &dL[k+1,k+1], &N, &L[k+1,k], &inc, &one, &dL[k+1,k], &inc)
        for i in range(N-k-1):
            dL[k+1+i,k] -= dL[k+1+i,k+1+i] * L[k+1+i,k]
            dL[k+1+i,k] /= L[k,k]
        dL[k,k] -= ddot(&n, &dL[k+1,k], &inc, &L[k+1,k], &inc)
        dL[k,k] /= 2. * L[k,k]

    symmetrize(dL)


### solve_triangular

# TODO get rid of argument b; this code doesn't depend on it, and some callers
# use that fact. get rid of other unused arguments too!

cdef inline void _solve_triangular_grad_arg0(
        double[::1,:] g, double[::1,:] ans,
        double[::1,:] a, double[::1,:] b, int trans,
        double[::1,:] out, double[::1,:] temp, double[::1,:] temp2) nogil:
    cdef int n = a.shape[0], nrhs = ans.shape[1]
    cdef double one = 1., neg1 = -1., zero = 0.
    cdef char _trans = 'N' if trans else 'T'
    cdef int info = 0

    copy(g, temp)
    dtrtrs('L', &_trans, 'N', &n, &nrhs, &a[0,0], &n, &temp[0,0], &n, &info)
    if trans:
        dgemm('N', 'T', &n, &n, &nrhs, &neg1, &ans[0,0], &n, &temp[0,0], &n, &zero, &temp2[0,0], &n)
    else:
        dgemm('N', 'T', &n, &n, &nrhs, &neg1, &temp[0,0], &n, &ans[0,0], &n, &zero, &temp2[0,0], &n)
    zero_upper_triangle(temp2)
    add_into(temp2, out)

cdef inline void _solve_triangular_grad_arg1(
        double[::1,:] g, double[::1,:] ans,
        double[::1,:] a, double[::1,:] b, int trans,
        double[::1,:] out, double[::1,:] temp) nogil:
    cdef int n = a.shape[0], nrhs = b.shape[1]
    cdef char _trans = 'N' if trans else 'T'
    cdef int info = 0

    copy(g, temp)
    dtrtrs('L', &_trans, 'N', &n, &nrhs, &a[0,0], &n, &temp[0,0], &n, &info)
    add_into(temp, out)

cdef inline void _solve_triangular_v_grad_arg0(
        double[::1] g, double[::1] ans,
        double[::1,:] a, double[::1] b, int trans,
        double[::1,:] out, double[::1] temp) nogil:
    cdef int n = a.shape[0], inc = 1, i, j
    cdef char _trans = 'N' if trans else 'T'
    cdef int info = 0

    copy_vector(g, temp)
    dtrtrs('L', &_trans, 'N', &n, &inc, &a[0,0], &n, &temp[0], &n, &info)
    if trans:
        for j in range(n):
            for i in range(j,n):
                out[i,j] -= ans[i] * temp[j]
    else:
        for j in range(n):
            for i in range(j,n):
                out[i,j] -= temp[i] * ans[j]

cdef inline void _solve_triangular_v_grad_arg1(
        double[::1] g, double[::1] ans,
        double[::1,:] a, double[::1] b, int trans,
        double[::1] out, double[::1] temp) nogil:
    cdef int n = a.shape[0], inc = 1
    cdef char _trans = 'N' if trans else 'T'
    cdef int info = 0

    copy_vector(g, temp)
    dtrtrs('L', &_trans, 'N', &n, &inc, &a[0,0], &n, &temp[0], &n, &info)
    add_into_vector(temp, out)

### dpotrs

cdef inline void _dpotrs_grad(
        double[::1,:] g,                       # input (outgrad)
        double[::1,:] ans,                     # result
        double[::1,:] L,                       # args subset
        double[::1,:] inter_ans,               # side info
        double[::1,:] g_L, double[::1,:] g_h,  # outputs
        double[::1,:] temp_nn, double[::1,:] temp_ns, double[::1,:] temp_ns2) nogil:
    _solve_triangular_grad_arg0(g, ans, L, inter_ans, 1, g_L, temp_ns2, temp_nn)
    zero_matrix(temp_ns)
    _solve_triangular_grad_arg1(g, ans, L, inter_ans, 1, temp_ns, temp_ns2)
    _solve_triangular_grad_arg0(temp_ns, inter_ans, L, temp_ns, 0, g_L, temp_ns2, temp_nn)
    _solve_triangular_grad_arg1(temp_ns, inter_ans, L, temp_ns, 0, g_h, temp_ns2)

cdef inline void _dpotrs_v_grad(
        double[::1] g,
        double[::1] ans,
        double[::1,:] L, double[::1] inter_ans,
        double[::1,:] g_L, double[::1] g_h,
        double[::1] temp_n, double[::1] temp_n2) nogil:
    zero_vector(temp_n)
    _solve_triangular_v_grad_arg0(g, ans, L, inter_ans, 1, g_L, temp_n2)
    _solve_triangular_v_grad_arg1(g, ans, L, inter_ans, 1, temp_n, temp_n2)
    _solve_triangular_v_grad_arg0(temp_n, inter_ans, L, temp_n, 0, g_L,temp_n2)
    _solve_triangular_v_grad_arg1(temp_n, inter_ans, L, temp_n, 0, g_h, temp_n2)
