from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from functools import partial
import cPickle as pickle

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.util import relu

from svae.recognition_models import mlp_recognize, init_mlp_recognize, \
    resnet_recognize, init_resnet_recognize, \
    linear_recognize, init_linear_recognize
from svae.forward_models import mlp_decode, mlp_loglike, init_mlp_loglike, \
    resnet_decode, resnet_loglike, init_resnet_loglike, \
    linear_decode, linear_loglike, init_linear_loglike

from svae.models.gmm_svae import run_inference, make_gmm_global_natparam, optimize_local_meanfield
from svae.lds import niw
from svae.hmm import dirichlet

recognize = resnet_recognize
loglike = resnet_loglike
decode = resnet_decode
init_recognize = init_resnet_recognize
init_loglike = init_resnet_loglike

normalize = lambda x: x / np.sum(x)

def encode_mean(data, natparam, psi):
    nn_potentials = recognize(data, psi)
    (_, gaussian_stats), _, _ = optimize_local_meanfield(natparam, nn_potentials)
    _, Ex, _, _ = gaussian_stats
    return Ex

def decode_mean(z, phi):
    mu, log_sigmasq = decode(z, phi)
    assert mu.ndim == 2 or mu.shape[1] == 1
    return mu.mean(axis=1)

def save(itr, data, params):
    with open('gmm_svae_synth_params.pkl', 'a') as outfile:
        pickle.dump(params, outfile, protocol=-1)

def plot(itr, axs, data, params):
    natparam, phi, psi = params
    ax0, ax1 = axs

    def generate_ellipse(mu, Sigma):
        t = np.hstack([np.arange(0, 2*np.pi, 0.01),0])
        circle = np.vstack([np.sin(t), np.cos(t)])
        ellipse = 2. * np.dot(np.linalg.cholesky(Sigma), circle)
        return ellipse[0] + mu[0], ellipse[1] + mu[1]

    def plot_or_update(idx, ax, x, y, alpha=1, **kwargs):
        if len(ax.lines) > idx:
            ax.lines[idx].set_data((x, y))
            ax.lines[idx].set_alpha(alpha)
        else:
            ax.plot(x, y, alpha=alpha, **kwargs)

    dir_hypers, all_niw_hypers = natparam
    weights = normalize(np.exp(dirichlet.expectedstats(dir_hypers)))
    components = map(niw.expected_standard_params, all_niw_hypers)

    latent_locations = encode_mean(data, natparam, psi)
    reconstruction = decode_mean(latent_locations, phi)

    ## make data-space plot

    # plot_or_update(0, ax0, reconstruction[:,0], reconstruction[:,1],
    #                color='b', marker='x', linestyle='')

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        x, y = generate_ellipse(mu, Sigma)
        transformed_x, transformed_y = decode_mean(np.vstack((x, y)).T, phi).T
        plot_or_update(idx, ax0, transformed_x, transformed_y,
                       alpha=min(1., num_clusters*weight), linestyle='-', linewidth=2)

    ## make latent space plot

    plot_or_update(0, ax1, latent_locations[:,0], latent_locations[:,1],
                   color='k', marker='o', linestyle='')

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        x, y = generate_ellipse(mu, Sigma)
        plot_or_update(idx+1, ax1, x, y, alpha=min(1., num_clusters*weight),
                       linestyle='-', linewidth=2)

    ax1.relim()
    ax1.autoscale_view(True, True, True)

    ## save plot

    plt.savefig('figures/running_gmm.png'.format(itr), dpi=150)
    # plt.savefig('figures/gmm_{:04d}.png'.format(itr), dpi=150, transparent=True)
    # plt.pause(0.0001)


def make_gmm_data():
    data = npr.permutation(np.concatenate(
        [-2. + npr.randn(50, P),
         2. + npr.randn(50, P),
         np.array([-3, 3.]) + npr.randn(50, P)]))
    data[:,1] *= 3.  # make eccentric
    return data

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


if __name__ == "__main__":
    npr.seed(1)
    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'orange', 'red', 'cyan', 'magenta', 'yellow'])))
    # plt.ion()

    K = 15  # number of components in mixture model
    N = 2   # number of latent dimensions
    P = 2   # number of observation dimensions

    ## generate synthetic data
    num_clusters = 5
    data = make_pinwheel_data(0.3, 0.05, num_clusters, 100, 0.25)

    # set prior natparam
    prior_natparam = make_gmm_global_natparam(K, N, alpha=0.1/K, niw_conc=0.5)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, recognize, loglike, prior_natparam)

    # set up plotting and callback
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(data[:,0], data[:,1], s=50, color='k', marker='+')
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].autoscale(False)
    axs[0].axis('off')
    axs[1].axis('off')
    fig.tight_layout()

    itr = 0
    def callback(_, vals, natgrad, params):
        global itr
        itr += 1
        print('{}: {}'.format(itr, np.mean(vals)))
        plot(itr, axs, data, params)
        # save(itr, data, params)

    ## instantiate optimizer
    optimize = adam(data, gradfun, callback)

    ## set initialization to something generic
    init_eta = make_gmm_global_natparam(K, N, alpha=1./10, niw_conc=2., random_scale=5.)
    init_phi = init_loglike([40, 40], N, P)
    init_psi = init_recognize([40, 40], N, P)
    params = init_eta, init_phi, init_psi

    ## optimize
    plot(0, axs, data, params)  # initial condition
    params = optimize(params, 10., 1e-2, num_epochs=1000, seq_len=50, num_samples=1)
