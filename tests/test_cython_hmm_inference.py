from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import primitive_with_aux

from svae.hmm.hmm_inference import hmm_logZ_python as python_hmm_logZ
from svae.hmm.cython_hmm_inference import hmm_logZ_normalized as cython_hmm_logZ_normalized, \
    hmm_logZ as cython_hmm_logZ, hmm_logZ_grad as cython_hmm_logZ_grad
from svae.util import allclose


cython_hmm_logZ = primitive_with_aux(cython_hmm_logZ)
def make_grad_hmm_logZ(intermediates, ans, hmm):
    _, pair_params, _ = hmm
    return lambda g: cython_hmm_logZ_grad(g, intermediates)
cython_hmm_logZ.defgrad(make_grad_hmm_logZ)


### util

def rand_hmm(n, T):
    return random_init_param(n), random_pair_param(n), random_node_potentials(n, T)

def random_init_param(n):
    return npr.randn(n)

def random_pair_param(n):
    return npr.randn(n, n)

def random_node_potentials(n, T):
    return npr.randn(T, n)


### tests

def test_lognorm():
    def compare_lognorms(hmm):
        py_logZ = python_hmm_logZ(hmm)
        cy_logZ = cython_hmm_logZ(hmm)
        cy_logZ2 = cython_hmm_logZ_normalized(hmm)[0]
        assert np.isclose(py_logZ, cy_logZ)
        assert np.isclose(py_logZ, cy_logZ2)

    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1, 10), npr.randint(10, 50)
        yield compare_lognorms, rand_hmm(n, T)


def test_lognorm_grad():
    def compare_lognorm_grads(hmm):
        dotter = npr.randn()
        py_grad = grad(lambda x: dotter * python_hmm_logZ(x))(hmm)
        cy_grad = grad(lambda x: dotter * cython_hmm_logZ(x))(hmm)
        assert allclose(py_grad, cy_grad)

    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1, 10), npr.randint(10, 50)
        yield compare_lognorm_grads, rand_hmm(n, T)
