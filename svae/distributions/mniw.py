from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import multigammaln, digamma
from autograd.scipy.linalg import solve_triangular
from autograd import grad
from autograd.util import make_tuple
from scipy.stats import chi2


def symmetrize(A):
    return (A + A.T)/2.


def is_posdef(A):
    return np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0.)


def standard_to_natural(nu, S, M, K):
    Kinv = np.linalg.inv(K)
    A = Kinv
    B = np.dot(Kinv, M.T)
    C = S + np.dot(M, B)
    d = nu
    return (A, B, C, d)


def natural_to_standard(natparam):
    A, B, C, d = natparam
    nu = d
    Kinv = A
    K = symmetrize(np.linalg.inv(Kinv))
    M = np.dot(K, B).T
    S = C - np.dot(M, B)
    return nu, S, M, K


def expectedstats_standard(nu, S, M, K, fudge=1e-8):
    m = M.shape[0]
    E_Sigmainv = nu*symmetrize(np.linalg.inv(S)) + fudge*np.eye(S.shape[0])
    E_Sigmainv_A = nu*np.linalg.solve(S, M)
    E_AT_Sigmainv_A = m*K + nu*symmetrize(np.dot(M.T, np.linalg.solve(S, M))) \
        + fudge*np.eye(K.shape[0])
    E_logdetSigmainv = digamma((nu-np.arange(m))/2.).sum() \
        + m*np.log(2) - np.linalg.slogdet(S)[1]

    assert is_posdef(E_Sigmainv)
    assert is_posdef(E_AT_Sigmainv_A)

    return make_tuple(
        -1./2*E_AT_Sigmainv_A, E_Sigmainv_A.T, -1./2*E_Sigmainv, 1./2*E_logdetSigmainv)


def logZ(natparam):
    nu, S, _, K = natural_to_standard(natparam)
    n = S.shape[0]
    return n*nu/2.*np.log(2) + multigammaln(nu/2., n) \
        - nu/2.*np.linalg.slogdet(S)[1] + n/2.*np.linalg.slogdet(K)[1]


def expectedstats(natparam):
    return expectedstats_standard(*natural_to_standard(natparam))


def expectedstats_autograd(natparam):
    return grad(logZ)(natparam)


def expected_standard_params(natparam):
    J11, J12, J22, _ = expectedstats(natparam)
    A = np.linalg.solve(-2.*J22, J12.T)
    Sigma = np.linalg.inv(-2*J22)
    return A, Sigma


def natural_sample(natparam):
    nu, S, M, K = natural_to_standard(natparam)

    def sample_invwishart(S, nu):
        n = S.shape[0]
        chol = np.linalg.cholesky(S)

        if (nu <= 81 + n) and (nu == np.round(nu)):
            x = npr.randn(nu, n)
        else:
            x = np.diag(np.sqrt(np.atleast_1d(chi2.rvs(nu - np.arange(n)))))
            x[np.triu_indices_from(x, 1)] = npr.randn(n*(n-1)//2)
        R = np.linalg.qr(x, 'r')
        T = solve_triangular(R.T, chol.T, lower=True).T
        return np.dot(T, T.T)

    def sample_mn(M, U, V):
        G = npr.normal(size=M.shape)
        return M + np.dot(np.dot(np.linalg.cholesky(U), G), np.linalg.cholesky(V).T)

    Sigma = sample_invwishart(S, nu)
    A = sample_mn(M, Sigma, K)

    return A, Sigma
