from __future__ import division
import numpy as np
import numpy.random as npr
from functools import partial

from svae.lds.mniw import expectedstats, expectedstats_autograd, \
    standard_to_natural, natural_to_standard, natural_sample
from test_util import rand_psd


def rand_mniw(p, n):
    S = rand_psd(p) + p*np.eye(p)
    K = rand_psd(n) + n*np.eye(n)
    M = npr.randn(p, n)
    nu = p + npr.uniform(1,3)
    return standard_to_natural(nu, S, M, K)


def test_param_conversion():
    npr.seed(0)

    def check_params(natparam):
        natparam2 = standard_to_natural(*natural_to_standard(natparam))
        assert all(map(np.allclose, natparam, natparam2))

    for _ in xrange(5):
        n, p = npr.randint(1,5), npr.randint(1,5)
        yield check_params, rand_mniw(n, p)


def test_expectedstats_autograd():
    npr.seed(0)

    def check_expectedstats(natparam):
        E_stats1 = expectedstats(natparam)
        E_stats2 = expectedstats_autograd(natparam)
        assert all(map(np.allclose, E_stats1, E_stats2))

    for _ in xrange(10):
        n, p = npr.randint(1,5), npr.randint(1,5)
        yield check_expectedstats, rand_mniw(n, p)

def test_expectedstats_montecarlo():
    npr.seed(0)
    N = 10000

    def montecarlo_stats(natparam):
        getstats = lambda A, Sigma: np.array([
            -1./2*A.T.dot(np.linalg.solve(Sigma,A)), np.linalg.solve(Sigma,A).T,
            -1./2*np.linalg.inv(Sigma), -1./2*np.linalg.slogdet(Sigma)[1]])
        mean = lambda x: sum(x) / len(x)
        return mean([getstats(*natural_sample(natparam)) for _ in xrange(N)])

    def check_expectedstats(natparam):
        E_stats1 = expectedstats(natparam)
        E_stats2 = montecarlo_stats(natparam)
        close = partial(np.allclose, atol=1e-1, rtol=1e-1)
        assert all(map(close, E_stats1, E_stats2))

    for _ in xrange(2):
        n, p = npr.randint(1,5), npr.randint(1,5)
        yield check_expectedstats, rand_mniw(n, p)
