from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from svae.distributions.niw import logZ, expectedstats, \
    standard_to_natural, natural_to_standard
from svae.util import rand_psd


def rand_niw(n):
    S = rand_psd(n) + n * np.eye(n)
    m = npr.randn(n)
    kappa = n + npr.uniform(1, 3)
    nu = n + npr.uniform(1, 3)
    return standard_to_natural(S, m, kappa, nu)

def rand_natparam(n, k):
    return np.squeeze(np.stack([rand_niw(n) for _ in range(k)]))

def test_param_conversion():
    npr.seed(0)

    def check_params(natparam):
        natparam2 = standard_to_natural(*natural_to_standard(natparam))
        assert np.allclose(natparam, natparam2)

    for _ in xrange(5):
        n, k = npr.randint(1, 5), npr.randint(1, 3)
        yield check_params, rand_natparam(n, k)

def test_expectedstats_autograd():
    npr.seed(0)

    def check_expectedstats(natparam):
        E_stats1 = expectedstats(natparam)
        E_stats2 = grad(logZ)(natparam)
        assert np.allclose(E_stats1, E_stats2)

    for _ in xrange(20):
        n, k = npr.randint(1, 5), npr.randint(1, 3)
        yield check_expectedstats, rand_natparam(n, k)
