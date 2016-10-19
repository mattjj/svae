from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from svae.distributions.niw import logZ, expectedstats, \
    standard_to_natural, natural_to_standard
from test_util import rand_psd


def rand_niw(n):
    S = rand_psd(n) + n * np.eye(n)
    m = npr.randn(n)
    kappa = n + npr.uniform(1, 3)
    nu = n + npr.uniform(1, 3)
    return standard_to_natural(S, m, kappa, nu)

def test_param_conversion():
    npr.seed(0)

    def check_params(natparam):
        natparam2 = standard_to_natural(*natural_to_standard(natparam))
        assert np.allclose(natparam, natparam2)

    for _ in xrange(5):
        n = npr.randint(1, 5)
        yield check_params, rand_niw(n)

def test_expectedstats_autograd():
    npr.seed(0)

    def check_expectedstats(natparam):
        E_stats1 = expectedstats(natparam)
        E_stats2 = grad(logZ)(natparam)
        assert np.allclose(E_stats1, E_stats2)

    for _ in xrange(10):
        n = npr.randint(1, 5)
        yield check_expectedstats, rand_niw(n)
