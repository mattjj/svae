from __future__ import division
import numpy as np
import numpy.random as npr

from svae.lds.lds_inference import natural_filter_forward_general


def rand_psd(n):
    a = npr.randn(n,n)
    return np.dot(a, a.T)

def make_symmetric_block(A, B, C):
    M = A.shape[0]
    out = np.zeros((2*M,2*M))
    out[:M,:M] = A
    out[:M,M:] = B
    out[M:,:M] = B.T
    out[M:,M:] = C
    return out

def to_dense(init_params, pair_params, node_params):
    T = node_params[0].shape[0]
    M = init_params[0].shape[0]

    # construct output
    J, h, extra_logZ_terms = np.zeros((T*M,T*M)), np.zeros(T*M), 0.

    # place init params
    J_init, h_init, negative_logZ_init = init_params
    J[:M,:M] += -2*J_init
    h[:M] += h_init
    extra_logZ_terms += negative_logZ_init

    # place pair params
    for t, pair_param in enumerate(pair_params):
        J_pair_11, J_pair_12, J_pair_22, negative_logZ_pair = pair_param
        J[t*M:(t+2)*M,t*M:(t+2)*M] += make_symmetric_block(-2*J_pair_11, -J_pair_12, -2*J_pair_22)
        extra_logZ_terms += negative_logZ_pair

    # place node params
    for t, node_param in enumerate(zip(*node_params)):
        J_node, h_node, negative_logZ_node = node_param
        J[t*M:(t+1)*M,t*M:(t+1)*M] += -2*J_node
        h[t*M:(t+1)*M] += h_node
        extra_logZ_terms += negative_logZ_node

    return J, h, extra_logZ_terms

def dense_lognorm(local_natparam, nn_potentials):
    init_params, pair_params = local_natparam
    node_params = nn_potentials

    J, h, extra_logZ_terms = to_dense(init_params, pair_params, node_params)
    lognorm = 1./2*np.dot(h, np.linalg.solve(J, h)) - 1./2*np.linalg.slogdet(J)[1]
    return lognorm + extra_logZ_terms

def lds_inference_lognorm(local_natparam, nn_potentials):
    init_params, pair_params = local_natparam
    _, lognorm = natural_filter_forward_general(init_params, pair_params, nn_potentials)
    return lognorm


### tests

def random_model(n, T):
    return random_local_natparam(n, T), random_nn_potentials(n, T)

def random_local_natparam(n, T):
    return random_init_param(n), [random_pair_param(n) for _ in range(T-1)]

def random_init_param(n):
    return -1./2*rand_psd(n), npr.randn(n), npr.randn()

def random_pair_param(n):
    J = rand_psd(2*n)
    J11 = -1./2*J[:n,:n]
    J12 = -J[:n,n:]
    J22 = -1./2*J[n:,n:]
    logZ = npr.randn()
    return J11, J12, J22, logZ

def random_nn_potentials(n, T):
    return map(np.array, zip(*[random_node_param(n) for _ in range(T)]))

def random_node_param(n):
    return -1./2*rand_psd(n), npr.randn(n), npr.randn()


def check_lognorm(local_natparam, nn_potentials):
    lognorm1 = dense_lognorm(local_natparam, nn_potentials)
    lognorm2 = lds_inference_lognorm(local_natparam, nn_potentials)
    assert np.isclose(lognorm1, lognorm2)

def test_lognorm():
    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1,5), npr.randint(1,10)
        local_natparam, nn_potentials = random_model(n, T)
        yield check_lognorm, local_natparam, nn_potentials
