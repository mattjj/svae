# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

from scipy.linalg.cython_blas cimport dsymv, ddot, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dtrtrs, dpotrf, dpotrs, dpotri
from libc.math cimport log

from svae.cython_util cimport *
from svae.cython_linalg_grads cimport *

cimport numpy as np  # TODO remove

### lognorm computation

# here, 'lognorm' refers to just the 1/2*np.dot(v,v) -
# np.sum(np.log(np.diag(L))) computation that happens inside natural_predict

cdef inline void _lognorm_grad_arg0(
        double g, double ans, double[::1,:] L, double[::1] v,
        double[::1,:] out) nogil:
    cdef int M = L.shape[0]
    cdef int i

    for i in range(M):
        out[i,i] += -g / L[i,i]

cdef inline void _lognorm_grad_arg1(
        double g, double ans, double[::1,:] L, double[::1] v,
        double[::1] out) nogil:
    cdef int M = v.shape[0]
    cdef int i

    for i in range(M):
        out[i] += g * v[i]


### natural_predict forward and backward computations

cdef inline double _natural_predict(
        # inputs
        double[::1, :] J, double[::1] h,
        double[::1,:] J11, double[::1,:] J12, double[::1,:] J22, double logZ,
        # outputs
        double[::1,:] J_predict, double[::1] h_predict,
        # temps (also outputs here)
        double[::1,:] L, double[::1] v, double[::1] v2, double[::1,:] temp,
        ) nogil:
    cdef int n = J.shape[0], inc = 1, i, j, info
    cdef double lognorm = 0., zero = 0., one = 1., neg1 = -1.

    # L = np.linalg.cholesky(J + J11)
    for j in range(n):
        for i in range(j, n):
            L[i,j] = J[i,j] + J11[i,j]
    dpotrf('L', &n, &L[0,0], &n, &info)

    # v = solve_triangular(L, h)
    copy_vector(h, v)
    dtrtrs('L', 'N', 'N', &n, &inc, &L[0,0], &n, &v[0], &n, &info)

    # lognorm = 1./2 * np.dot(v,v) - np.sum(np.log(np.diag(L)))
    for i in range(n):
        lognorm += v[i]**2
    lognorm /= 2.
    for i in range(n):
        lognorm -= log(L[i,i])

    # v2 = solve_triangular(L, v, 'T')
    copy_vector(v, v2)
    dtrtrs('L', 'T', 'N', &n, &inc, &L[0,0], &n, &v2[0], &n, &info)

    # h_predict = -np.dot(J12.T, v2)
    dgemv('T', &n, &n, &neg1, &J12[0,0], &n, &v2[0], &inc, &zero, &h_predict[0], &inc)

    # temp = solve_triangular(L, J12)
    copy(J12, temp)
    dtrtrs('L', 'N', 'N', &n, &n, &L[0,0], &n, &temp[0,0], &n, &info)

    # J_predict = J22 - np.dot(temp.T, temp)
    copy(J22, J_predict)
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp[0,0], &n, &temp[0,0], &n, &one, &J_predict[0,0], &n)

    return lognorm + logZ

cdef inline void _natural_predict_grad(
        # incoming gradients
        double [::1,:] g_J_predict, double[::1] g_h_predict, double g_lognorm,
        # temps from forward pass
        double[::1,:] J12, double[::1,:] L, double[::1] v, double[::1] v2, double[::1,:] temp,
        # outputs
        double[::1,:] g_J, double[::1] g_h,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2, double[::1,:] temp_nn3, double[::1,:] g_temp,
        double[::1] temp_n, double[::1] temp_n2, double[::1] g_v) nogil:
    cdef int n = g_h_predict.shape[0]
    cdef int inc = 1
    cdef double neg1 = -1., zero = 0., negtwo = -2.

    copy(g_J_predict, temp_nn)
    symmetrize(temp_nn)
    dgemm('N', 'N', &n, &n, &n, &negtwo, &temp[0,0], &n, &temp_nn[0,0], &n, &zero, &g_temp[0,0], &n)

    zero_matrix(temp_nn3)  # accumulator for g_L_i terms
    _solve_triangular_grad_arg0(g_temp, temp, L, J12, 0, temp_nn3, temp_nn, temp_nn2)

    zero_vector(g_v)
    dgemv('N', &n, &n, &neg1, &J12[0,0], &n, &g_h_predict[0], &inc, &zero, &temp_n2[0], &inc)
    _solve_triangular_v_grad_arg0(temp_n2, v2, L, v, 1, temp_nn3, temp_n)
    _solve_triangular_v_grad_arg1(temp_n2, v2, L, v, 1, g_v, temp_n)

    _lognorm_grad_arg0(g_lognorm, 0., L, v, temp_nn3)
    _lognorm_grad_arg1(g_lognorm, 0., L, v, g_v)

    _solve_triangular_v_grad_arg0(g_v, v, L, temp_n, 0, temp_nn3, temp_n)
    zero_vector(temp_n2)
    _solve_triangular_v_grad_arg1(g_v, v, L, temp_n, 0, temp_n2, temp_n)

    _cholesky_grad(L, temp_nn3)

    add_into(temp_nn3, g_J)
    add_into_vector(temp_n2, g_h)


### conditioning

cdef inline double _natural_condition_diag(
        # inputs
        double[::1,:] J, double[::1] h,
        double[::1] Jo, double[::1] ho, double logZ,
        # outputs
        double[::1,:] J_cond, double[::1] h_cond) nogil:
    cdef int M = J.shape[0]
    cdef int i, j

    for j in range(M):
        for i in range(M):
            J_cond[i,j] = J[i,j]

    for i in range(M):
        J_cond[i,i] += Jo[i]

    for i in range(M):
        h_cond[i] = h[i] + ho[i]

    return logZ

cdef inline void _natural_condition_diag_grad(
        # inputs (outgrads)
        double[::1,:] g_J_cond, double[::1] g_h_cond,
        # outputs
        double[::1,:] g_J, double[::1] g_h,
        double[::1] g_Jo, double[::1] g_ho) nogil:
    cdef int M = g_ho.shape[0]
    cdef int i, j

    for i in range(M):
        g_Jo[i] += g_J_cond[i,i]
        g_ho[i] += g_h_cond[i]
        g_h[i] += g_h_cond[i]

    for j in range(M):
        for i in range(M):
            g_J[i,j] += g_J_cond[i,j]


### lognorm from (J, h)

cdef inline double _natural_lognorm(
        double[::1,:] J, double[::1] h,
        double[::1,:] L, double[::1] v) nogil:
    cdef int n = J.shape[0], inc = 1, info, i
    cdef double lognorm = 0.

    # L = cholesky(J)
    copy(J, L)
    dpotrf('L', &n, &L[0,0], &n, &info)

    # v = solve_triangular(L, h)
    copy_vector(h, v)
    dtrtrs('L', 'N', 'N', &n, &inc, &L[0,0], &n, &v[0], &n, &info)

    # lognorm = 1./2 * np.dot(v,v) - np.sum(np.log(np.diag(L)))
    for i in range(n):
        lognorm += v[i]**2
    lognorm /= 2.
    for i in range(n):
        lognorm -= log(L[i,i])

    return lognorm

cdef inline void _natural_lognorm_grad(
        double g_lognorm,
        double[::1,:] g_J, double[::1] g_h,
        double[::1,:] L, double[::1] v,
        double[::1] g_v, double[::1] temp_n, double[::1,:] temp_nn) nogil:
    zero_vector(g_v)
    zero_matrix(temp_nn)
    _lognorm_grad_arg0(g_lognorm, 0., L, v, temp_nn)
    _lognorm_grad_arg1(g_lognorm, 0., L, v, g_v)

    _solve_triangular_v_grad_arg0(g_v, v, L, temp_n, 0, temp_nn, temp_n)
    _solve_triangular_v_grad_arg1(g_v, v, L, temp_n, 0, g_h, temp_n)

    _cholesky_grad(L, temp_nn)
    add_into(temp_nn, g_J)

### rts E-step operations

cdef inline void _info_to_mean(
        # inputs
        double[::1,:] J, double[::1] h,
        # outputs
        double[::1] mu, double[::1,:] Sigma) nogil:
    cdef int n = J.shape[0], info, inc = 1

    copy(J, Sigma)
    dpotrf('L', &n, &Sigma[0,0], &n, &info)
    copy_vector(h, mu)
    dpotrs('L', &n, &inc, &Sigma[0,0], &n, &mu[0], &n, &info)
    dpotri('L', &n, &Sigma[0,0], &n, &info)
    copy_lower_to_upper(Sigma)

cdef inline void _info_to_mean_grad(
        double[::1] g_mu, double[::1,:] g_Sigma,   # g
        double[::1] mu, double[::1,:] Sigma,       # ans
        double[::1,:] J, double[::1] h,            # args
        double[::1,:] g_J, double[::1] g_h,        # outputs
        double[::1,:] temp_nn, double[::1,:] temp_nn2) nogil:
    cdef int n = J.shape[0], inc = 1, i, j, info
    cdef double neg1 = -1., one = 1., zero = 0.

    for j in range(n):
        for i in range(n):
            g_Sigma[i,j] += g_mu[i] * h[j]

    dgemv('T', &n, &n, &one, &Sigma[0,0], &n, &g_mu[0], &inc, &one, &g_h[0], &inc)

    dgemm('T', 'N', &n, &n, &n, &neg1, &Sigma[0,0], &n, &g_Sigma[0,0], &n, &zero, &temp_nn[0,0], &n)
    dgemm('N', 'T', &n, &n, &n, &one, &temp_nn[0,0], &n, &Sigma[0,0], &n, &one, &g_J[0,0], &n)

cdef inline void _rts_backward_step(
        # inputs
        double[::1,:] Jns, double[::1] hns, double[::1] mun,
        double[::1,:] Jnp, double[::1] hnp,
        double[::1,:] Jf, double[::1] hf,
        double[::1,:] J11, double[::1,:] J12, double[::1,:] J22,
        # outputs
        double[::1,:] Js, double[::1] hs,
        double[::1] Ex, double [::1,:] ExxT, double[::1,:] ExnxT,
        # temps
        double[::1,:] L, double[::1,:] temp, double[::1] temp_n) nogil:
    cdef int i, j, n = Jns.shape[0], inc = 1, info
    cdef double one = 1., zero = 0., neg1 = -1.

    # L = chol(Jns - Jnp + J22)
    for j in range(n):
        for i in range(n):
            L[i,j] = Jns[i,j] - Jnp[i,j] + J22[i,j]
    dpotrf('L', &n, &L[0,0], &n, &info)

    # temp = solve_triangular(L, J12.T)
    for j in range(n):
        for i in range(n):
            temp[i,j] = J12[j,i]
    dtrtrs('L', 'N', 'N', &n, &n, &L[0,0], &n, &temp[0,0], &n, &info)

    # J11 = Jf + J11 (stored in Js)
    for j in range(n):
        for i in range(n):
            Js[i,j] = Jf[i,j] + J11[i,j]

    # Js = J11 - dot(temp.T, temp)
    dgemm('T', 'N', &n, &n, &n, &neg1, &temp[0,0], &n, &temp[0,0], &n, &one, &Js[0,0], &n)

    # temp_n = solve_triangular(L, hns - hnp)
    for i in range(n):
        temp_n[i] = hns[i] - hnp[i]
    dtrtrs('L', 'N', 'N', &n, &inc, &L[0,0], &n, &temp_n[0], &n, &info)

    # hs = hf - dot(temp.T, temp_n)
    copy_vector(hf, hs)
    dgemv('T', &n, &n, &neg1, &temp[0,0], &n, &temp_n[0], &inc, &one, &hs[0], &inc)

    # mu, sigma = info_to_mean(Js, hs, Ex, ExxT)  (stored in Ex and ExxT)
    _info_to_mean(Js, hs, Ex, ExxT)

    # ExnxT = -dpotrs(L, np.dot(J12.T, sigma)) + np.outer(mun, mu)
    dgemm('T', 'N', &n, &n, &n, &neg1, &J12[0,0], &n, &ExxT[0,0], &n, &zero, &ExnxT[0,0], &n)
    dpotrs('L', &n, &n, &L[0,0], &n, &ExnxT[0,0], &n, &info)
    for j in range(n):
        for i in range(n):
            ExnxT[i,j] += mun[i] * Ex[j]

    # ExxT = sigma + np.outer(mu, mu)
    for j in range(n):
        for i in range(n):
            ExxT[i,j] += Ex[i] * Ex[j]

cdef inline void _rts_backward_step_grad(
        # inputs
        double[::1,:] g_Js, double[::1] g_hs,  # updated inplace in step 2
        double[::1] g_Ex, double[::1,:] g_ExxT, double[::1,:] g_ExnxT,
        # ans
        double[::1,:] Js, double[::1] hs,
        double[::1] Ex, double[::1,:] ExxT, double[::1,:] ExnxT,
        # args
        double[::1,:] J12, double[::1] mun,
        # intermediates from forward pass
        double[::1,:] L, double[::1,:] temp, double[::1] temp_n,
        # outputs
        double[::1,:] g_Jns, double[::1] g_hns, double[::1] g_mun,
        double[::1,:] g_Jnp, double[::1] g_hnp,
        double[::1,:] g_Jf, double[::1] g_hf,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2, double[::1,:] temp_nn3,
        double[::1,:] temp_nn4, double[::1,:] temp_nn5, double[::1,:] temp_nn6,
        double[::1,:] g_L) nogil:
    cdef int i, j, n = L.shape[0]

    zero_upper_triangle(L)
    zero_matrix(g_L)
    _rts_3_grad(g_Ex, g_ExxT, g_ExnxT, Ex, ExnxT, J12, mun, L, g_L, g_mun,
                temp_nn, temp_nn2, temp_nn3, temp_nn4, temp_nn5, temp_nn6)
    _rts_2_grad(g_Ex, g_ExxT, Ex, ExxT, Js, hs, g_Js, g_hs,
                temp_nn, temp_nn2, temp_nn3)
    _rts_1_grad(g_Js, g_hs, L, temp, temp_n,
                g_L, g_hns, g_hnp, g_Jf, g_hf,
                temp_nn, temp_nn2, temp_nn3, temp_nn4)
    _cholesky_grad(L, g_L)
    for j in range(n):
        for i in range(n):
            g_Jns[i,j] += g_L[i,j]
            g_Jnp[i,j] -= g_L[i,j]

cdef inline void _rts_3_grad(
        # grads (some updated in-place)
        double[::1] g_Ex, double[::1,:] g_ExxT, double[::1,:] g_ExnxT,
        # ans (subset)
        double[::1] Ex, double[::1,:] ExnxT,
        # args (subset)
        double[::1,:] J12, double[::1] mun,
        # intermediates
        double [::1,:] L,
        # outputs (also update g_Ex and g_ExxT inplace)
        double[::1,:] g_L, double[::1] g_mun,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2, double[::1,:] temp_nn3,
        double[::1,:] temp_nn4, double[::1,:] temp_nn5, double[::1,:] temp_nn6) nogil:
    cdef int i, j, n = L.shape[0], inc = 1
    cdef double one = 1., zero = 0., neg1 = -1.

    dgemv('N', &n, &n, &one, &g_ExxT[0,0], &n, &Ex[0], &inc, &one, &g_Ex[0], &inc)
    dgemv('T', &n, &n, &one, &g_ExxT[0,0], &n, &Ex[0], &inc, &one, &g_Ex[0], &inc)

    dgemv('N', &n, &n, &one, &g_ExnxT[0,0], &n, &Ex[0], &inc, &one, &g_mun[0], &inc)
    dgemv('T', &n, &n, &one, &g_ExnxT[0,0], &n, &mun[0], &inc, &one, &g_Ex[0], &inc)

    # recompute temp_nn = Sigma_21 = ExnxT - np.outer(mun, mu)
    for j in range(n):
        for i in range(n):
            temp_nn[i,j] = ExnxT[i,j] - mun[i] * Ex[j]

    # recompute temp_nn2 = temp2 = solve_triangular(L, dot(-J12.T, ExxT - Ex Ex.T))
    # = np.dot(L.T, Sigma_21) (could use dsymm here)
    # note g_ExnxT is treated as g_Sigma21
    dgemm('T', 'N', &n, &n, &n, &one, &L[0,0], &n, &temp_nn[0,0], &n, &zero, &temp_nn2[0,0], &n)
    zero_matrix(temp_nn3)
    _dpotrs_grad(
        g_ExnxT, temp_nn, L, temp_nn2, g_L, temp_nn3,
        temp_nn4, temp_nn5, temp_nn6)
    dgemm('N', 'N', &n, &n, &n, &neg1, &J12[0,0], &n, &temp_nn3[0,0], &n, &one, &g_ExxT[0,0], &n)

cdef inline void _rts_2_grad(
        # grads
        double[::1] g_mu, double[::1,:] g_Sigma,
        # ans (subset)
        double[::1] Ex, double[::1,:] ExxT,
        # args
        double[::1,:] Js, double[::1] hs,
        # outputs
        double[::1,:] g_Js, double[::1] g_hs,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2, double[::1,:] temp_nn3) nogil:
    cdef int i, j, n = Ex.shape[0], info

    # recompute temp_nn = Sigma = ExxT - np.outer(Ex, Ex)
    for j in range(n):
        for i in range(n):
            temp_nn[i,j] = ExxT[i,j] - Ex[i] * Ex[j]
    _info_to_mean_grad(
        g_mu, g_Sigma, Ex, temp_nn, Js, hs,
        g_Js, g_hs,
        temp_nn2, temp_nn3)

cdef inline void _rts_1_grad(
        # inputs
        double[::1,:] g_Js, double[::1] g_hs,
        # intermediates
        double[::1,:] L, double[::1,:] temp, double[::1] temp_n,
        # outputs
        double[::1,:] g_L,
        double[::1] g_hns, double[::1] g_hnp,
        double[::1,:] g_Jf, double[::1] g_hf,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2, double[::1,:] temp_nn3,
        double[::1,:] temp_nn4) nogil:
    cdef int i, j, n = L.shape[0], inc = 1
    cdef double one = 1., zero = 0., neg1 = -1., neg2 = -2.

    dgemv('N', &n, &n, &neg1, &temp[0,0], &n, &g_hs[0], &inc, &zero, &temp_nn[0,0], &inc)
    for j in range(n):
        for i in range(n):
            temp_nn2[i,j] = -temp_n[i] * g_hs[j]
    add_into_vector(g_hs, g_hf)

    _solve_triangular_v_grad_arg0(temp_nn[:,0], temp_n, L, temp_n, 0, g_L, temp_nn3[:,0])
    _solve_triangular_v_grad_arg1(temp_nn[:,0], temp_n, L, temp_n, 0, g_hns, temp_nn3[:,0])
    for i in range(n):
        g_hnp[i] -= g_hns[i]

    copy(g_Js, temp_nn3)
    symmetrize(temp_nn3) # TODO maybe this is the extra symmetrize causing the problem?
    dgemm('N', 'N', &n, &n, &n, &neg2, &temp[0,0], &n, &temp_nn3[0,0], &n, &one, &temp_nn2[0,0], &n)

    add_into(g_Js, g_Jf)

    _solve_triangular_grad_arg0(temp_nn2, temp, L, temp_nn, 0, g_L, temp_nn3, temp_nn4)


### sampling

cdef inline void _natural_sample(
        # inputs
        double[::1,:] J, double[::1,:] h, double[::1,:] eps,
        # outputs
        double[::1,:] shaped_randvecs,
        # temps
        double[::1,:] L, double[::1,:] temp_ns) nogil:
    cdef int n = J.shape[0], nrhs = eps.shape[1], info
    cdef int i, j

    # L = np.linalg.cholesky(J)
    copy(J, L)
    dpotrf('L', &n, &L[0,0], &n, &info)

    # shaped_randvecs = solve_triangular(L, eps)
    copy(eps, shaped_randvecs)
    dtrtrs('L', 'T', 'N', &n, &nrhs, &L[0,0], &n, &shaped_randvecs[0,0], &n, &info)

    # shaped_randvecs += solve_posdef_from_cholesky(L, h)
    copy(h, temp_ns)
    dpotrs('L', &n, &nrhs, &L[0,0], &n, &temp_ns[0,0], &n, &info)
    for j in range(nrhs):
        for i in range(n):
            shaped_randvecs[i,j] += temp_ns[i,j]

cdef inline void _natural_sample_grad(
        # inputs (outgrad)
        double[::1,:] g_shaped_randvecs,
        # side info from forward pass
        # double[::1,:] L, double[::1,:] mu, double[::1,:] dpotrs_inter_ans, double[::1,:] zero_mean_randvecs,
        double[::1,:] J, double[::1,:] h, double[::1,:] L, double[::1,:] eps,
        # outputs
        double[::1,:] g_J, double[::1,:] g_h,
        # temps
        double[::1,:] temp_nn, double[::1,:] temp_nn2,
        double[::1,:] temp_ns, double[::1,:] temp_ns2, double[::1,:] temp_ns3) nogil:

    cdef int n = eps.shape[0], s = eps.shape[1], info

    zero_matrix(temp_nn2)  # accumulator for g_L
    # dpotrs_inter_ans = solve(L, h), stored in h
    dtrtrs('L', 'N', 'N', &n, &s, &L[0,0], &n, &h[0,0], &n, &info)
    # mu = dpotrs(L, h), stored in temp_ns3
    copy(h, temp_ns3)
    dtrtrs('L', 'T', 'N', &n, &s, &L[0,0], &n, &temp_ns3[0,0], &n, &info)
    _dpotrs_grad(
        g_shaped_randvecs, temp_ns3, L, h,
        temp_nn2, g_h,
        temp_nn, temp_ns, temp_ns2)
    # zero_mean_randvecs = solve(L, eps, 'T'), stored in eps
    dtrtrs('L', 'T', 'N', &n, &s, &L[0,0], &n, &eps[0,0], &n, &info)
    _solve_triangular_grad_arg0(
        g_shaped_randvecs, eps, L, temp_ns, 1,  # temp_ns just a placeholder
        temp_nn2,
        temp_ns, temp_nn)
    _cholesky_grad(L, temp_nn2)
    add_into(temp_nn2, g_J)

cdef inline double _natural_condition_on(
        # inputs
        double[::1,:] J, double[::1] h,
        double[::1,:] y, double[::1,:] Jxx, double[::1,:] Jxy,
        # outputs
        double[::1,:] J_cond, double[::1,:] h_cond) nogil:
    cdef int n = J.shape[0], s = y.shape[1]
    cdef int i, j
    cdef double zero = 0., neg1 = -1.

    # J_cond = J + Jxx
    for j in range(n):
        for i in range(n):
            J_cond[i,j] = J[i,j] + Jxx[i,j]

    # h_cond = h - dot(Jxy, y)  (note h acts like column vector, y.shape is (n, s))
    dgemm('N', 'N', &n, &s, &n, &neg1, &Jxy[0,0], &n, &y[0,0], &n, &zero, &h_cond[0,0], &n)
    for j in range(s):
        for i in range(n):
            h_cond[i,j] += h[i]

cdef inline void _natural_condition_on_grad(
        # inputs (outgrads)
        double[::1,:] g_J_cond, double[::1,:] g_h_cond,
        # outputs
        double[::1,:] g_J, double[::1] g_h, double[::1,:] g_y,
        # side info
        double[::1,:] Jxy) nogil:
    cdef int n = g_J.shape[0], s = g_y.shape[1], i, j
    cdef double one = 1., neg1 = -1.

    for j in range(s):
        for i in range(n):
            g_h[i] += g_h_cond[i,j]

    dgemm('T', 'N', &n, &s, &n, &neg1, &Jxy[0,0], &n, &g_h_cond[0,0], &n, &one, &g_y[0,0], &n)

    for j in range(n):
        for i in range(n):
            g_J[i,j] += g_J_cond[i,j]
