from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from svae.distributions.gaussian import logZ, expectedstats, \
    pack_dense, unpack_dense
from svae.util import rand_psd


def rand_gaussian(n):
    J = rand_psd(n) + n * np.eye(n)
    h = npr.randn(n)
    return pack_dense(-1./2*J, h)

def rand_natparam(n, k):
    return np.squeeze(np.stack([rand_gaussian(n) for _ in range(k)]))

def test_pack_dense():
    npr.seed(0)

    def check_params(natparam):
        natparam2 = pack_dense(*unpack_dense(natparam))
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
