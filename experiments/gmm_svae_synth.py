from __future__ import division, print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam, sgd
from svae.svae import make_gradfun
from svae.nnet import init_gresnet, make_loglike, gaussian_mean, gaussian_info
from svae.models.gmm import (run_inference, init_pgm_param, make_encoder_decoder,
                             dirichlet, niw)

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 10*npr.permutation(np.einsum('ti,tij->tj', features, rotations))

def make_plotter(encode_mean, decode_mean, data, num_clusters, plot_every=5):
    fig, (observation_axis, latent_axis) = plt.subplots(1, 2, figsize=(8,4))

    observation_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
    observation_axis.set_aspect('equal')
    observation_axis.autoscale(False)
    observation_axis.axis('off')
    latent_axis.set_aspect('equal')
    latent_axis.axis('off')
    fig.tight_layout()

    normalize = lambda arr: arr / np.sum(arr)

    def plot_ellipse(ax, weight, mean, cov):
        t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
        ax.plot(ellipse[0], ellipse[1], alpha=min(1., num_clusters*weight),
                linestyle='-', linewidth=2)

    def plot(i, val, params, grad):
        print('{}: {}'.format(i, val))
        if not i % plot_every:
            latent_axis.lines = []

            pgm_params, loglike_params, recogn_params = params
            x, y = encode_mean(data, pgm_params, recogn_params).T
            latent_axis.plot(x, y, color='k', marker='.', linestyle='')

            dirichlet_natparams, all_niw_natparams = pgm_params
            weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
            components = map(niw.expected_standard_params, all_niw_natparams)
            for weight, (mu, Sigma) in zip(weights, components):
                plot_ellipse(latent_axis, weight, mu, Sigma)

            plt.pause(0.1)

    return plot

if __name__ == "__main__":
    npr.seed(1)
    plt.ion()

    num_clusters = 3           # number of clusters in pinwheel data
    samples_per_cluster = 100  # number of samples per cluster in pinwheel
    K = 15                     # number of components in mixture model
    N = 2                      # number of latent dimensions
    P = 2                      # number of observation dimensions

    # generate synthetic data
    data = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

    # set prior natparam to something sparsifying but otherwise generic
    pgm_prior_params = init_pgm_param(K, N, alpha=0.1/K, niw_conc=0.5)

    # construct recognition and decoder networks and initialize them
    recognize, recogn_params = \
        init_gresnet(P, [(40, np.tanh), (40, np.tanh), (2*N, gaussian_info)])
    decode,   loglike_params = \
        init_gresnet(N, [(40, np.tanh), (40, np.tanh), (2*P, gaussian_mean)])
    loglike = make_loglike(decode)

    # set up plotting
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)
    plot = make_plotter(encode_mean, decode_mean, data, num_clusters, plot_every=5)

    # initialize gmm parameters
    pgm_params = init_pgm_param(K, N, alpha=1., niw_conc=1., random_scale=5.)
    params = pgm_params, loglike_params, recogn_params

    # instantiate svae gradient function
    gradfun = make_gradfun(run_inference, recognize, loglike, pgm_prior_params, data)

    # optimize
    params = sgd(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e3, callback=plot),
                 params, num_iters=1000, step_size=1e-3)
