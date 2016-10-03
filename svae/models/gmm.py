from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from itertools import repeat

from svae.util import contract, add, sub, unbox, shape
from svae.lds import niw, gaussian
from svae.hmm import dirichlet

normalize = lambda x: x / np.sum(x, 1, keepdims=True)


### GMM mean field inference functions

# TODO break out optimize local mean field loop?

def optimize_local_meanfield(global_natparam, node_potentials, tol=1e-3, max_iter=100):
    label_global, gaussian_global = global_natparam

    local_vlb = -np.inf
    label_stats = initialize_local_meanfield(label_global, node_potentials)
    for _ in xrange(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_vlb = \
            gaussian_meanfield(gaussian_global, unbox(node_potentials), label_stats)
        label_natparam, label_stats, label_vlb = \
            label_meanfield(label_global, gaussian_global, gaussian_stats)

        local_vlb, prev_local_vlb = label_vlb + gaussian_vlb, local_vlb
        if abs(local_vlb - prev_local_vlb) < tol:
            break
    else:
        print 'iteration limit reached'

    # recompute values that depend on node_potentials at optimum
    gaussian_natparam, gaussian_stats, gaussian_vlb = \
        gaussian_meanfield(gaussian_global, node_potentials, label_stats)
    label_natparam, label_stats, label_vlb = \
        label_meanfield(label_global, gaussian_global, gaussian_stats)

    stats = label_stats, gaussian_stats
    local_natparams = label_natparam, gaussian_natparam
    vlbs = label_vlb, gaussian_vlb

    return stats, local_natparams, vlbs

def initialize_local_meanfield(label_global, node_potentials):
    K = label_global.shape[0]
    T = node_potentials[0].shape[0]
    return normalize(npr.rand(T, K))

def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    partial_contract = lambda a, b: \
        sum(np.tensordot(x, y, axes=np.ndim(y)) for x, y in zip(a, b))

    gaussian_local_natparams = map(niw.expectedstats, gaussian_globals)
    node_params = np.array([
        partial_contract(gaussian_stats, natparam) for natparam in gaussian_local_natparams]).T

    local_natparam = dirichlet.expectedstats(label_global) + node_params
    stats = normalize(np.exp(local_natparam  - logsumexp(local_natparam, axis=1, keepdims=True)))
    vlb = np.sum(logsumexp(local_natparam, axis=1)) - contract(stats, node_params)

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

    def get_node_stats(gaussian_stats):
        ExxT, Ex, En, En = gaussian_stats
        return np.diagonal(ExxT, axis1=-1, axis2=-2), Ex, En

    local_natparam = get_local_natparam(gaussian_globals, node_potentials, label_stats)
    stats = gaussian.expectedstats(local_natparam)
    vlb = gaussian.logZ(local_natparam) \
        - contract(node_potentials, get_node_stats(stats)[:len(node_potentials)])

    return local_natparam, stats, vlb


### other inference functions used at optimum

def gaussian_sample(local_natparams, num_samples):
    from svae.lds.gaussian import natural_sample
    Js, hs = local_natparams[:2]
    return np.array([natural_sample(J, h, num_samples) for J, h in zip(Js, hs)])


### GMM global operations

def gmm_global_vlb(global_natparam, prior_natparam):
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


### prior initialization

def init_pgm_param(K, N, alpha, niw_conc=10., random_scale=0.):
    def make_label_global_natparam(k, random):
        return alpha * np.ones(k) if not random else alpha + npr.rand(k)

    def make_gaussian_global_natparam(n, random):
        nu, S, mu, kappa = n+niw_conc, (n+niw_conc)*np.eye(n), np.zeros(n), niw_conc
        mu = mu + random_scale * npr.randn(*mu.shape)
        return niw.standard_to_natural(nu, S, mu, kappa)

    label_global_natparam = make_label_global_natparam(K, random_scale > 0)
    gaussian_global_natparams = [make_gaussian_global_natparam(N, random_scale > 0) for _ in xrange(K)]

    return label_global_natparam, gaussian_global_natparams


### inference functions

def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    label_global_natparam, gaussian_global_natparams = global_natparam

    (label_stats, gaussian_stats), (_, gaussian_local_natparam), (label_vlb, gaussian_vlb) = \
        optimize_local_meanfield(global_natparam, nn_potentials)

    stats = get_global_stats(label_stats, gaussian_stats)
    samples = gaussian_sample(gaussian_local_natparam, num_samples)
    local_kl = -label_vlb - gaussian_vlb
    global_kl = -gmm_global_vlb(global_natparam, prior_natparam)

    return samples, unbox(stats), global_kl, local_kl

def make_encoder_decoder(recognize, decode):
    def encode_mean(data, natparam, recogn_params):
        nn_potentials = recognize(recogn_params, data)
        (_, gaussian_stats), _, _ = optimize_local_meanfield(natparam, nn_potentials)
        _, Ex, _, _ = gaussian_stats
        return Ex

    def decode_mean(z, phi):
        mu, _ = decode(z, phi)
        return mu.mean(axis=1)

    return encode_mean, decode_mean


### plotting util for 2D

def make_plotter_2d(recognize, decode, data, num_clusters, params, plot_every):
    import matplotlib.pyplot as plt
    if data.shape[1] != 2: raise ValueError, 'make_plotter_2d only works with 2D data'

    fig, (observation_axis, latent_axis) = plt.subplots(1, 2, figsize=(8,4))
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)

    observation_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
    observation_axis.set_aspect('equal')
    observation_axis.autoscale(False)
    observation_axis.axis('off')
    latent_axis.set_aspect('equal')
    latent_axis.axis('off')
    fig.tight_layout()

    def plot_encoded_means(ax, params):
        pgm_params, loglike_params, recogn_params = params
        encoded_means = encode_mean(data, pgm_params, recogn_params)
        if isinstance(ax, plt.Axes):
            ax.plot(encoded_means[:,0], encoded_means[:,1], color='k', marker='.', linestyle='')
        elif isinstance(ax, plt.Line2D):
            ax.set_data(encoded_means.T)
        else:
            raise ValueError

    def plot_ellipse(ax, alpha, mean, cov, line=None):
        t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
        if line:
            line.set_data(ellipse)
            line.set_alpha(alpha)
        else:
            ax.plot(ellipse[0], ellipse[1], alpha=alpha, linestyle='-', linewidth=2)

    def plot_components(ax, params):
        pgm_params, loglike_params, recogn_params = params
        dirichlet_natparams, all_niw_natparams = pgm_params
        normalize = lambda arr: arr / np.sum(arr) * num_clusters
        weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
        components = map(niw.expected_standard_params, all_niw_natparams)
        lines = repeat(None) if isinstance(ax, plt.Axes) else ax
        for weight, (mu, Sigma), line in zip(weights, components, lines):
            plot_ellipse(ax, weight, mu, Sigma, line)

    def plot(i, val, params, grad):
        print('{}: {}'.format(i, val))
        if (i % plot_every) == (-1 % plot_every):
            plot_encoded_means(latent_axis.lines[0], params)
            plot_components(latent_axis.lines[1:], params)
            plt.pause(0.1)

    plot_encoded_means(latent_axis, params)
    plot_components(latent_axis, params)
    plt.pause(0.1)

    return plot
