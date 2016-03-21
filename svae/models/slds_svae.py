from __future__ import division, print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import make_tuple
from functools import partial
import sys

from svae.svae import make_gradfun
from svae.util import add, sub, scale, randn_like, contract, unbox, make_binop, tuplify, shape

from svae.lds.lds_inference import \
    cython_natural_lds_inference_general as natural_lds_inference_general, \
    cython_natural_lds_estep_general as natural_lds_estep_general, \
    cython_natural_lds_sample as natural_lds_sample
from svae.lds import niw, mniw, gaussian
from svae.hmm.hmm_inference import hmm_estep, hmm_logZ
from svae.hmm import dirichlet
from lds_svae import lds_prior_expectedstats


### global natural parameter making

# the slds global natural parameters comprise hmm prior natural parameters and a set
# of lds prior natural parameters

def make_slds_global_natparam(num_states, state_dim, alpha=5., sticky_bias=0., random=False):
    hmm_global_natparam = make_hmm_global_natparam(num_states, alpha, sticky_bias, random=random)
    lds_global_natparams = make_lds_global_natparams(num_states, state_dim, random=random)

    return hmm_global_natparam, lds_global_natparams


# the hmm prior natural parameters are a pair: dirichlet (dir) parameters for
# the prior on the initial state distribution and a set of dirichlet parameters
# (mdir) for the prior on the transition matrix

def make_hmm_global_natparam(num_states, alpha=1., sticky_bias=0., random=False):
    def make_dir_natparam(num_states):
        return alpha*np.ones(num_states) if not random else alpha+npr.rand(num_states)

    def make_mdir_natparam(num_states):
        return sticky_bias * np.eye(num_states) + \
            np.array([make_dir_natparam(num_states) for _ in range(num_states)])

    dir_natparam = make_dir_natparam(num_states)
    mdir_natparam = make_mdir_natparam(num_states)

    return dir_natparam, mdir_natparam


# the lds prior natural parameters are a pair: normal inverse wishart (niw)
# natural parameters for each initial state distribution and and matrix normal
# inverse wishart (mniw) natural parameters for each dynamics distribution

def make_lds_global_natparams(num_states, state_dim, random=False):
    def make_niw_natparam(n):
        if not random:
            nu, S, mu, kappa = n+10., (n+10.)*np.eye(n), np.zeros(n), 10.
        else:
            nu, S, mu, kappa = n+4.+npr.rand(), (n+npr.rand())*np.eye(n), npr.randn(n), npr.rand()
        return niw.standard_to_natural(nu, S, mu, kappa)

    def make_mniw_natparam(n):
        if not random:
            nu, S, M, K = n+10., (n+10.)*np.eye(n), np.zeros((n, n)), 10.*np.eye(n)
        else:
            nu, S, M, K = n+4.+npr.rand(), (n+npr.rand())*np.eye(n), \
                1e-2*npr.randn(n, n), (n+npr.rand())*np.eye(n)
        return mniw.standard_to_natural(nu, S, M, K)

    niw_natparams = [make_niw_natparam(state_dim) for _ in range(num_states)]
    mniw_natparams = [make_mniw_natparam(state_dim) for _ in range(num_states)]

    return zip(niw_natparams, mniw_natparams)


### lds

def lds_meanfield(lds_global_natparams, node_potentials, hmm_expected_stats):
    local_natparam = get_var_lds_local_natparam(lds_global_natparams, hmm_expected_stats)
    vlb, expected_stats = natural_lds_estep_general(local_natparam, node_potentials)
    return local_natparam, expected_stats, vlb


def get_all_lds_local_natparams(lds_global_natparams):
    lds_params = map(lds_prior_expectedstats, lds_global_natparams)
    init_params, pair_params = zip(*lds_params)
    return init_params, pair_params


def get_var_lds_local_natparam(lds_global_natparams, hmm_expected_stats):
    _, _, expected_states = hmm_expected_stats
    all_init_params, all_pair_params = get_all_lds_local_natparams(lds_global_natparams)

    dense_init_params = map(np.stack, zip(*all_init_params))
    dense_pair_params = map(np.stack, zip(*all_pair_params))

    contract = lambda a: lambda b: np.tensordot(a, b, axes=1)
    init_param = map(contract(expected_states[0]), dense_init_params)
    pair_params = map(contract(expected_states[1:]), dense_pair_params)

    return init_param, pair_params


### hmm

def hmm_meanfield(hmm_global_natparam, lds_global_natparam, lds_expected_stats):
    init_params, pair_params = get_hmm_local_natparam(hmm_global_natparam)
    node_params = get_arhmm_local_nodeparams(lds_global_natparam, lds_expected_stats)

    local_natparam = init_params, pair_params, node_params
    vlb, stats = hmm_estep(local_natparam)

    return local_natparam, stats, vlb


def get_hmm_local_natparam(hmm_global_natparam):
    return hmm_prior_expectedstats(hmm_global_natparam)


def hmm_prior_expectedstats(natparam):
    dir_natparam, mdir_natparam = natparam

    init_params = dirichlet.expectedstats(dir_natparam)
    pair_params = np.array(map(dirichlet.expectedstats, mdir_natparam))

    return init_params, pair_params


def get_arhmm_local_nodeparams(lds_global_natparam, lds_expected_stats):
    init_stats, pair_stats = lds_expected_stats[:2]
    all_init_params, all_pair_params = get_all_lds_local_natparams(lds_global_natparam)

    dense_init_params = map(np.stack, zip(*all_init_params))
    dense_pair_params = map(np.stack, zip(*all_pair_params))

    partial_contract = lambda a: lambda b: contract(a, b)
    init_node_potential = np.array(map(partial_contract(init_stats), all_init_params))

    partial_contract = lambda a: lambda b: \
        sum(np.tensordot(x, y, axes=np.ndim(y)) for x, y in zip(a,b))
    remaining_node_potentials = np.vstack(map(partial_contract(pair_stats), all_pair_params)).T

    node_potentials = np.vstack((init_node_potential, remaining_node_potentials))

    return node_potentials


def get_hmm_vlb(lds_global_natparam, hmm_local_natparam, lds_expected_stats):
    init_params, pair_params, _ = hmm_local_natparam
    node_params = get_arhmm_local_nodeparams(lds_global_natparam, lds_expected_stats)
    local_natparam = make_tuple(init_params, pair_params, node_params)
    return hmm_logZ(local_natparam)


### slds local mean field

def optimize_local_meanfield(global_natparam, node_potentials, tol=1e-2):
    hmm_global, lds_global = global_natparam

    lds_stats = initialize_local_meanfield(node_potentials)
    local_vlb = -np.inf

    for _ in xrange(100):
        hmm_natparam, hmm_stats, hmm_vlb = hmm_meanfield(hmm_global, lds_global, lds_stats)
        lds_natparam, lds_stats, lds_vlb = lds_meanfield(lds_global, node_potentials, hmm_stats)

        local_vlb, prev_local_vlb = hmm_vlb + lds_vlb, local_vlb
        if abs(local_vlb - prev_local_vlb) < tol:
            break
    else:
        print('iteration limit reached', file=sys.stderr)

    return (hmm_stats, lds_stats), (hmm_natparam, lds_natparam), (hmm_vlb, lds_vlb)


def optimize_local_meanfield_withlabels(global_natparam, node_potentials, labels, tol=1e-2):
    hmm_global, lds_global = global_natparam
    N = hmm_global[0].shape[0]

    def count_transitions(labels):
        return np.vstack(
            [np.bincount(labels[1:][labels[:-1] == i], minlength=N) for i in xrange(N)])

    def indicators(labels):
        return np.eye(N)[labels]

    normalize = lambda a: a / a.sum(1, keepdims=True)

    def get_hmm_stats(labels):
        init_stats = indicators(labels[0])
        pair_stats = count_transitions(labels)
        node_stats = normalize(indicators(labels) + 1e-2)
        return init_stats, pair_stats, node_stats

    hmm_stats = get_hmm_stats(labels)
    lds_natparam, lds_stats, lds_vlb = lds_meanfield(lds_global, node_potentials, hmm_stats)

    return (hmm_stats, lds_stats), (None, lds_natparam), (0., lds_vlb)


def initialize_local_meanfield(node_potentials):
    # TODO maybe i should initialize the hmm states instead of the lds states...

    def compute_stats(sampled_states):
        out = np.outer
        get_init_stats = lambda x0: (out(x0, x0), x0, 1., 1.)
        get_pair_stats = lambda x1, x2: (out(x1, x1), out(x1, x2), out(x2, x2), 1.)

        init_stats = get_init_stats(sampled_states[0])
        pair_stats = map(get_pair_stats, sampled_states[:-1], sampled_states[1:])

        return init_stats, map(np.array, zip(*pair_stats))

    # construct random walk natparam
    N = node_potentials[0].shape[1]
    A = 0.9 * np.eye(N)
    init_params = -1./2*np.eye(N), np.zeros(N), 0.
    pair_params = -1./2*np.dot(A.T, A), A.T, -1./2*np.eye(N), 0.
    natparam = init_params, pair_params

    sampled_states = np.squeeze(natural_lds_sample(natparam, node_potentials, num_samples=1))
    init_stats, pair_stats = compute_stats(sampled_states)

    return init_stats, pair_stats


def get_global_stats(hmm_stats, lds_stats):
    def get_hmm_global_stats(hmm_stats):
        return hmm_stats[:-1]

    def get_lds_global_stats(hmm_stats, lds_stats):
        _, _, expected_states = hmm_stats
        init_stats, pair_stats = lds_stats

        contract = lambda w: lambda p: np.tensordot(w, p, axes=1)
        global_init_stats = tuple(scale(w, init_stats) for w in expected_states[0])
        global_pair_stats = tuple(map(contract(w), pair_stats) for w in expected_states[1:].T)

        return zip(global_init_stats, global_pair_stats)

    return get_hmm_global_stats(hmm_stats), get_lds_global_stats(hmm_stats, lds_stats)


### slds global vlb terms

def slds_prior_vlb(global_natparam, prior_natparam):
    expected_stats = slds_prior_expectedstats(global_natparam)
    return contract(sub(prior_natparam, global_natparam), expected_stats) \
        - (slds_prior_logZ(prior_natparam) - slds_prior_logZ(global_natparam))


def slds_prior_logZ(natparam):
    hmm_prior_natparam, lds_prior_natparams = natparam

    def lds_prior_logZ(natparam):
        niw_natparam, mniw_natparam = natparam
        return niw.logZ(niw_natparam) + mniw.logZ(mniw_natparam)

    def hmm_prior_logZ(natparam):
        dirichlet_natparams, mdirichlet_natparams = natparam
        return dirichlet.logZ(dirichlet_natparams) \
            + sum(map(dirichlet.logZ, mdirichlet_natparams))

    hmm_prior_term = hmm_prior_logZ(hmm_prior_natparam)
    lds_prior_term = sum(map(lds_prior_logZ, lds_prior_natparams))

    return hmm_prior_term + lds_prior_term


def slds_prior_expectedstats(global_natparam):
    hmm_natparam, lds_natparams = global_natparam

    def lds_prior_expectedstats(natparam):
        niw_natparam, mniw_natparam = natparam
        return niw.expectedstats(niw_natparam), mniw.expectedstats(mniw_natparam)

    def hmm_prior_expectedstats(natparam):
        dirichlet_natparams, mdirichlet_natparams = natparam
        return dirichlet.expectedstats(dirichlet_natparams), \
            dirichlet.expectedstats(mdirichlet_natparams)  # broadcasts on rows

    return hmm_prior_expectedstats(hmm_natparam), map(lds_prior_expectedstats, lds_natparams)


### combined inference function for svae

def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    hmm_global_natparam, lds_global_natparam = global_natparam

    # optimize local mean field (can use unboxed val for low-level code)
    (hmm_stats, _), (hmm_local_natparam, lds_local_natparam), _ = \
        optimize_local_meanfield(global_natparam, unbox(nn_potentials))

    # recompute terms that depend on nn_potentials at optimum (using boxed val)
    samples, lds_stats, lds_normalizer = natural_lds_inference_general(
        lds_local_natparam, nn_potentials, num_samples)
    hmm_vlb = get_hmm_vlb(lds_global_natparam, hmm_local_natparam, lds_stats)

    # get global statistics from the local expected stats
    global_lds_stats, local_lds_stats = lds_stats[:-1], lds_stats[-1]
    expected_stats = get_global_stats(hmm_stats, global_lds_stats)

    # compute global and local vlb terms
    global_vlb = slds_prior_vlb(global_natparam, prior_natparam)
    lds_vlb = lds_normalizer - contract(nn_potentials, local_lds_stats)
    local_vlb = hmm_vlb + lds_vlb

    return samples, expected_stats, global_vlb, local_vlb


def run_inference_withlabels(prior_natparam, global_natparam, potentials_and_labels, num_samples):
    nn_potentials, labels = potentials_and_labels
    hmm_global_natparam, lds_global_natparam = global_natparam

    # optimize local mean field (can use unboxed val for low-level code)
    (hmm_stats, _), (hmm_local_natparam, lds_local_natparam), _ = \
        optimize_local_meanfield_withlabels(global_natparam, unbox(nn_potentials), labels)

    # recompute terms that depend on nn_potentials at optimum (using boxed val)
    samples, lds_stats, lds_normalizer = natural_lds_inference_general(
        lds_local_natparam, nn_potentials, num_samples)

    # get global statistics from the local expected stats
    global_lds_stats, local_lds_stats = lds_stats[:-1], lds_stats[-1]
    expected_stats = get_global_stats(hmm_stats, global_lds_stats)

    # compute global and local vlb terms
    global_vlb = slds_prior_vlb(global_natparam, prior_natparam)
    lds_vlb = lds_normalizer - contract(nn_potentials, local_lds_stats)
    local_vlb = lds_vlb

    return samples, expected_stats, global_vlb, local_vlb
