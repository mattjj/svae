from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal as mvn

from svae.lds.gaussian import natural_predict, condition_on

from test_util import rand_psd

npr.seed(0)


### tests

def test_condition_on():
    def rand_model(p, n):
        return npr.randn(n), rand_psd(n), npr.randn(p, n), npr.randn(p), rand_psd(p)

    def condition_on_2(mu_x, sigma_x, A, y, sigma_obs):
        sigma_xy = sigma_x.dot(A.T)
        sigma_yy = A.dot(sigma_x).dot(A.T) + sigma_obs
        mu = mu_x + sigma_xy.dot(np.linalg.solve(sigma_yy, y - A.dot(mu_x)))
        sigma = sigma_x - sigma_xy.dot(np.linalg.solve(sigma_yy, sigma_xy.T))
        ll = mvn.logpdf(y, A.dot(mu_x), sigma_yy)
        return (mu, sigma), ll

    def check_condition_on(p, n):
        model = rand_model(p, n)
        (mu, sigma), ll = condition_on(*model)
        (mu2, sigma2), ll2 = condition_on_2(*model)

        assert np.allclose(mu, mu2)
        assert np.allclose(sigma, sigma2)
        assert np.isclose(ll, ll2)

    for _ in xrange(10):
        yield check_condition_on, npr.randint(1, 10), npr.randint(1, 10)


def test_natural_predict():
    def rand_model(n):
        J, h = rand_psd(n), npr.randn(n)
        bigJ = rand_psd(2*n)
        J11, J12, J22 = bigJ[:n,:n], bigJ[:n,n:], bigJ[n:,n:]
        logZ = npr.randn()
        return -1./2*J, h, -1./2*J11, -J12, -1./2*J22, logZ

    def natural_predict_nochol(J, h, J11, J12, J22, logZ):
        # convert from natural parameter to the usual J definitions
        J, J11, J12, J22 = -2*J, -2*J11, -J12, -2*J22

        Jnew = J + J11
        Jpredict = J22 - J12.T.dot(np.linalg.solve(Jnew,J12))
        hpredict = -J12.T.dot(np.linalg.solve(Jnew,h))
        lognorm = -1./2*np.linalg.slogdet(Jnew)[1] + 1./2*h.dot(np.linalg.solve(Jnew,h))
        return (-1./2*Jpredict, hpredict), lognorm + logZ

    def check_natural_predict(n):
        model = rand_model(n)
        (J1, h1), term1 = natural_predict(*model)
        (J2, h2), term2 = natural_predict_nochol(*model)

        assert np.allclose(J1, J2)
        assert np.allclose(h1, h2)
        assert np.allclose(term1, term2)

    for _ in xrange(10):
        yield check_natural_predict, npr.randint(1, 10)
