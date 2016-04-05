from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp

from svae.util import contract, add, sub, unbox, shape
from svae.lds import niw, gaussian
from svae.hmm import dirichlet


### GMM mean field inference functions

def optimize_local_meanfield(global_natparam, node_potentials, tol=1e-3, max_iter=100):
    def initialize_local_meanfield(label_global, gaussian_global, node_potentials):
        K = label_global.shape[0]
        T = node_potentials[0].shape[0]
        return npr.rand(T, K)

    label_global, gaussian_global = global_natparam
    label_stats = initialize_local_meanfield(label_global, gaussian_global, node_potentials)

    local_vlb = -np.inf
    for _ in xrange(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_vlb = \
            gaussian_meanfield(gaussian_global, node_potentials, label_stats)
        label_natparam, label_stats, label_vlb = \
            label_meanfield(label_global, gaussian_global, gaussian_stats)

        local_vlb, prev_local_vlb = label_vlb + gaussian_vlb, local_vlb
        if abs(local_vlb - prev_local_vlb) < tol:
            break
    else:
        print 'iteration limit reached'

    stats = label_stats, gaussian_stats
    local_natparams = label_natparam, gaussian_natparam
    vlbs = label_vlb, gaussian_vlb

    return stats, local_natparams, vlbs

def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    gaussian_local_natparams = map(niw.expectedstats, gaussian_globals)
    partial_contract = lambda a, b: \
        sum(np.tensordot(x, y, axes=np.ndim(y)) for x, y, in zip(a, b))
    node_params = np.array([
        partial_contract(gaussian_stats, natparam) for natparam in gaussian_local_natparams]).T

    local_natparam = dirichlet.expectedstats(label_global) + node_params
    stats = np.exp(local_natparam  - logsumexp(local_natparam, axis=1, keepdims=True))
    vlb = np.sum(logsumexp(local_natparam, axis=1))

    return local_natparam, stats, vlb

def gaussian_meanfield(gaussian_globals, node_potentials, label_stats):
    def make_full_potentials(node_potentials):
        Jdiag, h = node_potentials[:2]
        T, N = h.shape
        return Jdiag[...,None] * np.eye(N)[None,...], h, np.zeros(T), np.zeros(T)

    def get_local_natparam(gaussian_globals, node_potentials, label_stats):
        local_natparams = [np.tensordot(label_stats, param, axes=1)
                           for param in zip(*map(niw.expectedstats, gaussian_globals))]
        return add(local_natparams, make_full_potentials(node_potentials))

    local_natparam = get_local_natparam(gaussian_globals, node_potentials, label_stats)
    stats = gaussian.expectedstats(local_natparam)
    vlb = gaussian.logZ(local_natparam)

    return local_natparam, stats, vlb


### other inference functions used at optimum

def gaussian_sample(local_natparams, node_potentials, num_samples):
    from svae.lds.gaussian import natural_sample
    return np.array([
        natural_sample(J + np.diag(Jo), h + ho, num_samples)
        for (J, h), (Jo, ho) in zip(zip(*local_natparams[:2]), zip(*node_potentials[:2]))])

### GMM global operations

def gmm_prior_vlb(global_natparam, prior_natparam):
    def gmm_prior_logZ(natparam):
        dir_natparam, niw_natparams = natparam
        return dirichlet.logZ(dir_natparam) + sum(map(niw.logZ, niw_natparams))

    def gmm_prior_expectedstats(natparam):
        dir_natparam, niw_natparams = natparam
        return dirichlet.expectedstats(dir_natparam), map(niw.expectedstats, niw_natparams)

    expected_stats = gmm_prior_expectedstats(global_natparam)
    return contract(sub(prior_natparam, global_natparam), expected_stats) \
        - (gmm_prior_logZ(prior_natparam) - gmm_prior_logZ(global_natparam))

def get_global_stats(label_stats, gaussian_stats):
    contract = lambda w: lambda p: np.tensordot(w, p, axes=1)
    global_label_stats = np.sum(label_stats, axis=0)
    global_gaussian_stats = tuple(map(contract(w), gaussian_stats) for w in label_stats.T)
    return global_label_stats, global_gaussian_stats

def get_local_gaussian_stats(gaussian_stats):
    ExxT, Ex, En, En = gaussian_stats
    return np.diagonal(ExxT, axis1=-1, axis2=-2), Ex, En

def make_gmm_global_natparam(K, N, alpha, random=False):
    def make_label_global_natparam(k, random):
        return alpha * np.ones(k) if not random else alpha + npr.rand(k)

    def make_gaussian_global_natparam(n, random):
        if not random:
            nu, S, mu, kappa = n+10., (n+10.)*np.eye(n), np.zeros(n), 10.
        else:
            nu, S, mu, kappa = n+4.+npr.rand(), (n+npr.rand())*np.eye(n), npr.randn(n), npr.rand()
        return niw.standard_to_natural(nu, S, mu, kappa)

    label_global_natparam = make_label_global_natparam(K, random)
    gaussian_global_natparams = [make_gaussian_global_natparam(N, random) for _ in xrange(K)]

    return label_global_natparam, gaussian_global_natparams


### inference function

def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    label_global_natparam, gaussian_global_natparams = global_natparam

    # optimize local mean field (using unboxed val)
    (label_stats, _), (_, gaussian_local_natparam), _ = \
        optimize_local_meanfield(global_natparam, unbox(nn_potentials))

    # recompute values that depend on nn_potentials at optimum
    _, gaussian_stats, gaussian_normalizer = \
        gaussian_meanfield(gaussian_global_natparams, nn_potentials, label_stats)
    _, _, label_vlb = \
        label_meanfield(label_global_natparam, gaussian_global_natparams, gaussian_stats)
    local_gaussian_stats = get_local_gaussian_stats(gaussian_stats)

    # compute samples of gaussian values
    samples = gaussian_sample(gaussian_local_natparam, nn_potentials, num_samples)

    # get global statistics from the local expected stats
    expected_stats = get_global_stats(label_stats, gaussian_stats)

    # compute global and local vlb terms
    gaussian_vlb = gaussian_normalizer - contract(nn_potentials, local_gaussian_stats)
    local_vlb = label_vlb + gaussian_vlb
    global_vlb = gmm_prior_vlb(global_natparam, prior_natparam)

    return samples, expected_stats, global_vlb, local_vlb
