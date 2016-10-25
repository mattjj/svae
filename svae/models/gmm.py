from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from itertools import repeat
from functools import partial

from svae.util import unbox, getval, flat, normalize
from svae.distributions import dirichlet, categorical, niw, gaussian

### inference functions for the SVAE interface

def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    _, stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    samples = gaussian.natural_sample(local_natparam[1], num_samples)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl

def make_encoder_decoder(recognize, decode):
    def encode_mean(data, natparam, recogn_params):
        nn_potentials = recognize(recogn_params, data)
        (_, gaussian_stats), _, _, _ = local_meanfield(natparam, nn_potentials)
        _, Ex, _, _ = gaussian.unpack_dense(gaussian_stats)
        return Ex

    def decode_mean(z, phi):
        mu, _ = decode(z, phi)
        return mu.mean(axis=1)

    return encode_mean, decode_mean

### GMM prior on \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)

def init_pgm_param(K, N, alpha, niw_conc=10., random_scale=0.):
    def init_niw_natparam(N):
        nu, S, m, kappa = N+niw_conc, (N+niw_conc)*np.eye(N), np.zeros(N), niw_conc
        m = m + random_scale * npr.randn(*m.shape)
        return niw.standard_to_natural(S, m, kappa, nu)

    dirichlet_natparam = alpha * (npr.rand(K) if random_scale else np.ones(K))
    niw_natparam = np.stack([init_niw_natparam(N) for _ in range(K)])

    return dirichlet_natparam, niw_natparam

def prior_logZ(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    return dirichlet.logZ(dirichlet_natparam) + niw.logZ(niw_natparams)

def prior_expectedstats(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    dirichlet_expectedstats = dirichlet.expectedstats(dirichlet_natparam)
    niw_expectedstats = niw.expectedstats(niw_natparams)
    return dirichlet_expectedstats, niw_expectedstats

def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expectedstats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    logZ_difference = prior_logZ(global_natparam) - prior_logZ(prior_natparam)
    return np.dot(natparam_difference, expected_stats) - logZ_difference

### GMM mean field functions

def local_meanfield(global_natparam, node_potentials):
    dirichlet_natparam, niw_natparams = global_natparam
    node_potentials = gaussian.pack_dense(*node_potentials)

    # compute expected global parameters using current global factors
    label_global = dirichlet.expectedstats(dirichlet_natparam)
    gaussian_globals = niw.expectedstats(niw_natparams)

    # compute mean field fixed point using unboxed node_potentials
    label_stats = meanfield_fixed_point(label_global, gaussian_globals, getval(node_potentials))

    # compute values that depend directly on boxed node_potentials at optimum
    gaussian_natparam, gaussian_stats, gaussian_kl = \
        gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
    label_natparam, label_stats, label_kl = \
        label_meanfield(label_global, gaussian_globals, gaussian_stats)

    # collect sufficient statistics for gmm prior (sum across conditional iid)
    dirichlet_stats = np.sum(label_stats, 0)
    niw_stats = np.tensordot(label_stats, gaussian_stats, [0, 0])

    local_stats = label_stats, gaussian_stats
    prior_stats = dirichlet_stats, niw_stats
    natparam = label_natparam, gaussian_natparam
    kl = label_kl + gaussian_kl

    return local_stats, prior_stats, natparam, kl

def meanfield_fixed_point(label_global, gaussian_globals, node_potentials, tol=1e-3, max_iter=100):
    kl = np.inf
    label_stats = initialize_meanfield(label_global, node_potentials)
    for i in xrange(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_kl = \
            gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
        label_natparam, label_stats, label_kl = \
            label_meanfield(label_global, gaussian_globals, gaussian_stats)

        kl, prev_kl = label_kl + gaussian_kl, kl
        if abs(kl - prev_kl) < tol:
            break
    else:
        print 'iteration limit reached'

    return label_stats

def gaussian_meanfield(gaussian_globals, node_potentials, label_stats):
    global_potentials = np.tensordot(label_stats, gaussian_globals, [1, 0])
    natparam = node_potentials + global_potentials
    stats = gaussian.expectedstats(natparam)
    kl = np.tensordot(node_potentials, stats, 3) - gaussian.logZ(natparam)
    return natparam, stats, kl

def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    node_potentials = np.tensordot(gaussian_stats, gaussian_globals, [[1,2], [1,2]])
    natparam = node_potentials + label_global
    stats = categorical.expectedstats(natparam)
    kl = np.tensordot(stats, node_potentials) - categorical.logZ(natparam)
    return natparam, stats, kl

def initialize_meanfield(label_global, node_potentials):
    T, K = node_potentials.shape[0], label_global.shape[0]
    return normalize(npr.rand(T, K))

### plotting util for 2D

def make_plotter_2d(recognize, decode, data, num_clusters, params, plot_every):
    import matplotlib.pyplot as plt
    if data.shape[1] != 2: raise ValueError, 'make_plotter_2d only works with 2D data'

    fig, (observation_axis, latent_axis) = plt.subplots(1, 2, figsize=(8,4))
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)

    observation_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
    observation_axis.set_aspect('equal')
    observation_axis.autoscale(False)
    latent_axis.set_aspect('equal')
    observation_axis.axis('off')
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

    def get_component(niw_natparam):
        neghalfJ, h, _, _ = gaussian.unpack_dense(niw_natparam)
        J = -2 * neghalfJ
        return np.linalg.solve(J, h), np.linalg.inv(J)

    def plot_components(ax, params):
        pgm_params, loglike_params, recogn_params = params
        dirichlet_natparams, niw_natparams = pgm_params
        normalize = lambda arr: np.minimum(1., arr / np.sum(arr) * num_clusters)
        weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
        components = map(get_component, niw.expectedstats(niw_natparams))
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
