from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.stats import norm

from svae.forward_models import _diagonal_gaussian_loglike


def check_diag_gauss_loglike(x, mu, log_sigmasq):
    loglike = _diagonal_gaussian_loglike(x, mu, log_sigmasq)
    scipy_loglike = np.mean(norm.logpdf(x[:,None,:], mu, np.sqrt(np.exp(log_sigmasq))), axis=1).sum()
    assert np.isclose(loglike, scipy_loglike)


def test_diag_gauss_loglike():
    npr.seed(0)
    for _ in xrange(50):
        T, K, p = npr.randint(1, 20), npr.randint(1, 5), npr.randint(1, 10)
        yield check_diag_gauss_loglike, npr.randn(T, p), npr.randn(T, K, p), npr.randn(T, K, p)
