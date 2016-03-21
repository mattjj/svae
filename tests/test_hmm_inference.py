from __future__ import division
import numpy as np
import numpy.random as npr

from svae.hmm import hmm_inference
from svae.util import allclose


### parameter makers

def make_hmm_natparam(num_states, T):
    row_normalize = lambda a: a / np.sum(a, axis=1, keepdims=True)

    init_param = np.log(npr.rand(num_states))
    pair_param = np.log(row_normalize(npr.rand(num_states, num_states)))
    node_potentials = np.log(npr.rand(T, num_states))

    return init_param, pair_param, node_potentials


### check expected statistics code

def check_hmm_estep(natparam):
    vlb1, stats1 = hmm_inference.hmm_estep(natparam)
    vlb2, stats2 = hmm_inference.hmm_estep_slow(natparam)

    assert np.isclose(vlb1, vlb2)
    assert allclose(stats1, stats2)


def test_hmm_estep():
    npr.seed(0)
    for _ in xrange(3):
        yield check_hmm_estep, make_hmm_natparam(npr.randint(2,5), npr.randint(5,10))
