from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from svae.distributions.dirichlet import logZ, expectedstats


def rand_dirichlet(n):
    return 10.*npr.rand(n)

def rand_natparam(n, k):
    return np.squeeze(np.stack([rand_dirichlet(n) for _ in range(k)]))

def test_expectedstats_autograd():
    npr.seed(0)

    def check_expectedstats(natparam):
        E_stats1 = expectedstats(natparam)
        E_stats2 = grad(logZ)(natparam)
        assert np.allclose(E_stats1, E_stats2)

    for _ in xrange(20):
        n, k = npr.randint(1, 5), npr.randint(1, 3)
        yield check_expectedstats, rand_natparam(n, k)
