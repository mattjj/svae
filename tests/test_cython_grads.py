from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.linalg import solve_triangular as _solve_triangular
from autograd import grad

from svae.lds.gaussian import natural_predict as _natural_predict, \
    natural_sample as _natural_sample, \
    natural_rts_backward_step
from svae.util import solve_triangular, solve_posdef_from_cholesky, \
    contract, allclose, shape

from test_cython_linalg_grads import solve_triangular_grad_arg0, \
    solve_triangular_grad_arg1, cholesky_grad, dpotrs_grad
from test_cython_gaussian_grads import natural_predict as __natural_predict, \
    natural_predict_grad as _natural_predict_grad, \
    natural_sample as _natural_sample, \
    natural_sample_grad, rts_backward_step, natural_lognorm_grad, \
    lognorm_grad_arg0, lognorm_grad_arg1, info_to_mean_grad, \
    rts_3_grad, rts_1_grad, rts_backward_step_grad


## util!

def randn_like(x):
    return npr.RandomState(0).randn(*np.shape(x))

def to_scalar(x):
    return np.sum(x * randn_like(x))

def rand_psd(n):
    a = npr.randn(n,n)
    return np.dot(a, a.T)

al2d = lambda x: x if x.ndim > 1 else x[...,None]

def check(a, b):
    assert np.allclose(a, b)


## solve_triangular grad functions

def test_solve_triangular_grads():
    npr.seed(0)
    n = 3

    foo = lambda L, v, trans: to_scalar(solve_triangular(L, v, trans))
    L = np.linalg.cholesky(rand_psd(n))

    for v in [npr.randn(n), npr.randn(n,n)]:
        for trans in ['N', 'T']:
            ans = solve_triangular(L, v, trans)

            a = grad(foo, 0)(L, v, trans)
            b = solve_triangular_grad_arg0(randn_like(ans), ans, L, v, trans)
            check(a, b)

            a = grad(foo, 1)(L, v, trans)
            b = solve_triangular_grad_arg1(randn_like(ans), ans, L, v, trans)
            check(a, b)


## lognorm grad functions

def test_lognorm_grads():
    npr.seed(0)
    n = 3

    L = np.linalg.cholesky(rand_psd(n))
    v = npr.randn(n)

    foo = lambda L, v: 1./2*np.dot(v,v) - np.sum(np.log(np.diag(L)))
    ans = foo(L, v)

    a = grad(foo, 0)(L, v)
    b = lognorm_grad_arg0(1., ans, L, v)
    check(a, b)

    a = grad(foo, 1)(L, v)
    b = lognorm_grad_arg1(1., ans, L, v)
    check(a, b)


### natural predict grad functions

# a version of natural_predict that has (J, h) as its first argument
def natural_predict(belief, J11, J12, J22, logZ):
    J, h = belief
    return _natural_predict(J, h, J11, J12, J22, logZ)


def natural_predict_forward_temps(J, J11, J12, h):
    J, J11, J12 = -2*J, -2*J11, -J12
    L = np.linalg.cholesky(J + J11)
    v = solve_triangular(L, h)
    lognorm = 1./2*np.dot(v,v) - np.sum(np.log(np.diag(L)))
    v2 = solve_triangular(L, v, trans='T')
    temp = solve_triangular(L, J12)
    return L, v, v2, temp, h, lognorm


def natural_predict_grad(g, ans, belief, J11, J12, J22, logZ):
    J, h = belief
    (g_J_predict, g_h_predict), g_lognorm = g

    # re-run forward pass (need to pass these things back here!)
    L, v, v2, temp, h, lognorm = natural_predict_forward_temps(J, J11, J12, h)
    J12 = -J12

    # run the backward pass
    # NEEDS: L, v, v2, temp
    # ALSO USES: h, lognorm
    g_temp = np.dot(temp, (g_J_predict + g_J_predict.T)/2.)
    g_L_1 = solve_triangular_grad_arg0(g_temp, temp, L, J12, 'N')

    g_a = -np.dot(J12, g_h_predict)
    g_L_2 = solve_triangular_grad_arg0(g_a, v2, L, v, 'T')
    g_v_1 = solve_triangular_grad_arg1(g_a, v2, L, v, 'T')

    g_L_3 = lognorm_grad_arg0(g_lognorm, lognorm, L, v)
    g_v_2 = lognorm_grad_arg1(g_lognorm, lognorm, L, v)

    g_L_4 = solve_triangular_grad_arg0(g_v_1 + g_v_2, v, L, h, 'N')
    # print 'high-level: {}'.format((g_v_1 + g_v_2, v, L))
    g_h   = solve_triangular_grad_arg1(g_v_1 + g_v_2, v, L, h, 'N')
    # print 'high_level: {}'.format(g_h)

    g_J = cholesky_grad(L, g_L_1 + g_L_2 + g_L_3 + g_L_4)

    return (-2*g_J, g_h)


def test_natural_predict_grad():
    npr.seed(0)
    n = 3

    J = rand_psd(n)
    h = npr.randn(n)
    bigJ = rand_psd(2*n)
    J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]
    logZ = npr.randn()
    J, J11, J12, J22 = -1./2*J, -1./2*J11, -J12, -1./2*J22

    ans = natural_predict((J, h), J11, J12, J22, logZ)
    dotter = (randn_like(J), randn_like(h)), randn_like(1.)

    def foo(*args):
        (J, h), logZ = natural_predict(*args)
        (a, b), c = dotter
        return np.sum(a*J) + np.sum(b*h) + c*logZ

    result1 = grad(foo)((J, h), J11, J12, J22, logZ)
    result2 = natural_predict_grad(dotter, ans, (J, h), J11, J12, J22, logZ)

    L, v, v2, temp, _, _ = natural_predict_forward_temps(J, J11, J12, h)
    result3 = _natural_predict_grad(dotter[0][0], dotter[0][1], dotter[1], -J12, L, v, v2, temp)

    for a, b in zip(result1, result2):
        check(a, b)

    for a, b in zip(result2, result3):
        check(a, b)


def test_natural_predict():
    npr.seed(0)
    n = 3

    J = rand_psd(n)
    h = npr.randn(n)
    bigJ = rand_psd(2*n)
    J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]
    logZ = npr.randn()
    J, J11, J12, J22 = -1./2*J, -1./2*J11, -J12, -1./2*J22

    (J_pred_1, h_pred_1), lognorm1 = _natural_predict(J, h, J11, J12, J22, logZ)
    (J_pred_2, h_pred_2), lognorm2 = __natural_predict(J, h, J11, J12, J22, logZ)

    assert np.allclose(J_pred_1, J_pred_2)
    assert np.allclose(h_pred_1, h_pred_2)
    assert np.isclose(lognorm1, lognorm2)


### the other nontrivial function in filter grad

def test_natural_lognorm_grad():
    npr.seed(0)
    n = 3

    J = rand_psd(n)
    h = npr.randn(n)

    def natural_lognorm((J, h)):
        L = np.linalg.cholesky(J)
        v = solve_triangular(L, h)
        return 1./2*np.dot(v, v) - np.sum(np.log(np.diag(L)))

    g_J_1, g_h_1 = grad(lambda x: np.pi*natural_lognorm(x))((J, h))

    L = np.linalg.cholesky(J)
    v = solve_triangular(L, h)
    g_J_2, g_h_2 = natural_lognorm_grad(np.pi, L, v)

    assert np.allclose(g_J_1, g_J_2)
    assert np.allclose(g_h_1, g_h_2)


### functions for backward sampling

def test_dpotrs_grad():
    npr.seed(0)
    n = 3
    s = 5

    J = rand_psd(n)
    h = npr.randn(n, s)
    L = np.linalg.cholesky(J)
    dpotrs = lambda (L, h): solve_triangular(L, solve_triangular(L, h), 'T')
    ans = dpotrs((L, h))

    dotter = npr.randn(*ans.shape)

    assert np.allclose(ans, np.linalg.solve(J, h))

    g_L_1, g_h_1 = grad(lambda x: np.sum(dotter * dpotrs(x)))((L, h))
    g_L_2, g_h_2 = dpotrs_grad(dotter, ans, L, h)

    assert np.allclose(g_L_1, g_L_2)
    assert np.allclose(g_h_1, g_h_2)

def test_natural_sample():
    npr.seed(0)
    n = 3
    s = 5

    J = rand_psd(n)
    h = npr.randn(n, s)
    eps = npr.randn(n, s)

    def natural_sample(J, h, eps):
        mu = np.linalg.solve(J, h)
        L = np.linalg.cholesky(J)
        noise = solve_triangular(L, eps, 'T')
        return mu + noise

    sample1 = natural_sample(J, h, eps)
    sample2 = _natural_sample(J, h, eps)

    assert np.allclose(sample1, sample2)

def test_natural_sample_grad():
    npr.seed(0)
    n = 3
    s = 5

    J = rand_psd(n)
    h = npr.randn(n, s)
    eps = npr.randn(n, s)

    dotter = npr.randn(*eps.shape)

    def natural_sample(J, h, eps):
        L = np.linalg.cholesky(J)
        mu = solve_posdef_from_cholesky(L, h)
        noise = solve_triangular(L, eps, 'T')
        return mu + noise

    g_J_1, g_h_1 = grad(lambda (J, h): np.sum(dotter * natural_sample(J, h, eps)))((J, h))
    g_J_2, g_h_2 = natural_sample_grad(dotter, natural_sample(J, h, eps), J, h, eps)

    assert np.allclose(g_J_1, g_J_2)
    assert np.allclose(g_h_1, g_h_2)


### rts backward step

def test_rts_backward_step():
    npr.seed(0)
    n = 3

    Jns = rand_psd(n)
    hns = npr.randn(n)
    mun = npr.randn(n)

    Jnp = rand_psd(n)
    hnp = npr.randn(n)

    Jf = rand_psd(n) + 10*np.eye(n)
    hf = npr.randn(n)

    bigJ = rand_psd(2*n)
    J11, J12, J22 = -1./2*bigJ[:n,:n], -bigJ[:n,n:], -1./2*bigJ[n:,n:]

    next_smooth = -1./2*Jns, hns, mun
    next_pred = -1./2*Jnp, hnp
    filtered = -1./2*Jf, hf

    pair_param = J11, J12, J22, 0.

    Js1, hs1, (mu1, ExxT1, ExxnT1) = natural_rts_backward_step(
        next_smooth, next_pred, filtered, pair_param)
    Js2, hs2, (mu2, ExxT2, ExnxT2) = rts_backward_step(
        next_smooth, next_pred, filtered, pair_param)

    assert np.allclose(Js1, Js2)
    assert np.allclose(hs1, hs2)

    assert np.allclose(mu1, mu2)
    assert np.allclose(ExxT1, ExxT2)
    assert np.allclose(ExxnT1, ExnxT2)

def test_info_to_mean_grad():
    npr.seed(0)
    n = 3

    g_mu = npr.randn(n)
    g_Sigma = npr.randn(3, 3)

    J = rand_psd(n)
    h = npr.randn(n)

    def info_to_mean((J, h)):
        Sigma = np.linalg.inv(J)
        mu = np.dot(Sigma, h)
        return mu, Sigma

    def fun1((J, h)):
        mu, Sigma = info_to_mean((J, h))
        return np.sum(g_mu * mu) + np.sum(g_Sigma * Sigma)

    g_J_1, g_h_1 = grad(fun1)((J, h))
    g_J_2, g_h_2 = info_to_mean_grad(g_mu, g_Sigma, J, h)

    assert np.allclose(g_h_1, g_h_2)
    assert np.allclose(g_J_1, g_J_2)


def test_rts_3():
    npr.seed(0)
    n = 3

    # inputs
    L = np.linalg.cholesky(rand_psd(n))
    Sigma = rand_psd(n)
    mu = npr.randn(n)
    mun = npr.randn(n)

    # constants
    J12 = rand_psd(2*n)[:n,n:]

    # outgrads
    g_ExnxT = npr.randn(n,n)
    g_ExxT  = npr.randn(n,n)
    g_Ex    = npr.randn(n)

    def step3(L, Sigma, mu, mun):
        temp2 = np.dot(-J12.T, Sigma)
        Sigma_21 = solve_posdef_from_cholesky(L, temp2)

        ExnxT = Sigma_21 + np.outer(mun, mu)
        ExxT = Sigma + np.outer(mu, mu)

        return mu, ExxT, ExnxT

    # ans
    Ex, ExxT, ExnxT = step3(L, Sigma, mu, mun)

    # compare grads
    def fun(args):
        Ex, ExxT, ExnxT = step3(*args)
        return np.sum(ExnxT * g_ExnxT) + np.sum(ExxT * g_ExxT) + np.sum(Ex * g_Ex)

    g_L1, g_Sigma1, g_mu1, g_mun1 = grad(fun)((L, Sigma, mu, mun))
    g_L2, g_Sigma2, g_mu2, g_mun2 = rts_3_grad(
            g_Ex, g_ExxT, g_ExnxT,
            Ex, ExxT, ExnxT,
            L, Sigma, mu, mun,
            J12)

    assert np.allclose(g_L1, g_L2)
    assert np.allclose(g_Sigma1, g_Sigma2)
    assert np.allclose(g_mu1, g_mu2)
    assert np.allclose(g_mun1, g_mun2)

def test_rts_1():
    npr.seed(0)
    n = 3

    # inputs
    Jns = rand_psd(n) + 10*np.eye(n)
    hns = npr.randn(n)
    Jnp = rand_psd(n)
    hnp = npr.randn(n)
    Jf = rand_psd(n)
    hf = npr.randn(n)

    # constants
    bigJ = rand_psd(2*n)
    J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]
    L = np.linalg.cholesky(Jns - Jnp + J22)

    # outgrads
    g_Js = npr.randn(n,n)
    g_hs = npr.randn(n)

    def step1(L, hns, hnp, Jf, hf):
        temp = solve_triangular(L, J12.T)
        Js = Jf + J11 - np.dot(temp.T, temp)
        hs = hf -  np.dot(temp.T, solve_triangular(L, hns - hnp))
        return Js, hs

    # ans
    Js, hs = step1(L, hns, hnp, Jf, hf)

    def fun(args):
        Js, hs = step1(*args)
        return np.sum(g_Js * Js) + np.sum(g_hs * hs)

    g_L1, g_hns1, g_hnp1, g_Jf1, g_hf1 = grad(fun)((L, hns, hnp, Jf, hf))
    g_L2, g_hns2, g_hnp2, g_Jf2, g_hf2 = rts_1_grad(
        g_Js, g_hs,
        Js, hs,
        L, hns, hnp, Jf, hf,
        J11, J12)

    assert np.allclose(g_hns1, g_hns2)
    assert np.allclose(g_hnp1, g_hnp2)
    assert np.allclose(g_Jf1, g_Jf2)
    assert np.allclose(g_hf1, g_hf2)
    assert np.allclose(g_L1, g_L2)

def test_rts_backward_step_grad():
    npr.seed(0)
    n = 5

    Jns = rand_psd(n) + 10*np.eye(n)
    hns = npr.randn(n)
    mun = npr.randn(n)

    Jnp = rand_psd(n)
    hnp = npr.randn(n)

    Jf = (rand_psd(n) + 10*np.eye(n))
    hf = npr.randn(n)

    bigJ = rand_psd(2*n)
    J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]

    next_smooth = Jns, hns, mun
    next_pred = Jnp, hnp
    filtered = Jf, hf

    pair_param = J11, J12, J22, 0.

    dotter = g_Js, g_hs, (g_Ex, g_ExxT, g_ExnxT) = \
        npr.randn(n,n), npr.randn(n), (npr.randn(n), npr.randn(n,n), npr.randn(n,n))

    # this function wraps natural_rts_backward_step to take care of factors of 2
    def fun(next_smooth, next_pred, filtered, pair_param):
        (Jns, hns, mun), (Jnp, hnp), (Jf, hf) = next_smooth, next_pred, filtered
        next_smooth, next_pred, filtered = (-1./2*Jns, hns, mun), (-1./2*Jnp, hnp), (-1./2*Jf, hf)

        J11, J12, J22, logZ_pair = pair_param
        pair_param = -1./2*J11, -J12, -1./2*J22, logZ_pair

        neghalfJs, hs, (Ex, ExxT, ExnxT) = natural_rts_backward_step(
            next_smooth, next_pred, filtered, pair_param)

        Js = -2*neghalfJs

        return Js, hs, (Ex, ExxT, ExnxT)

    # ans
    Js, hs, (Ex, ExxT, ExnxT) = fun(next_smooth, next_pred, filtered, pair_param)

    def gfun(next_smooth, next_pred, filtered):
        vals = fun(next_smooth, next_pred, filtered, pair_param)
        assert shape(vals) == shape(dotter)
        return contract(dotter, vals)

    g1 = grad(lambda x: gfun(*x))((next_smooth, next_pred, filtered))
    g2 = rts_backward_step_grad(
        g_Js, g_hs, g_Ex, g_ExxT, g_ExnxT,
        next_smooth, next_pred, filtered, pair_param,
        Js, hs, (Ex, ExxT, ExnxT))

    assert allclose(g1, g2)

if __name__ == "__main__":
    # main method here for doing timing comparisons

    npr.seed(0)
    n = 3

    J = rand_psd(n)
    h = npr.randn(n)
    bigJ = rand_psd(2*n)
    J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]
    logZ = npr.randn()
    J, J11, J12, J22 = -1./2*J, -1./2*J11, -J12, -1./2*J22

    ans = natural_predict((J, h), J11, J12, J22, logZ)
    dotter = (randn_like(J), randn_like(h)), randn_like(1.)

    def foo(*args):
        (J, h), logZ = natural_predict(*args)
        (a, b), c = dotter
        return np.sum(a*J) + np.sum(b*h) + c*logZ

    result1 = grad(foo)((J, h), J11, J12, J22, logZ)

    L, v, v2, temp, _, _ = natural_predict_forward_temps(J, J11, J12, h)
    result3 = _natural_predict_grad(dotter[0][0], dotter[0][1], dotter[1], -J12, L, v, v2, temp)
