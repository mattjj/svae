from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import linear_recognize, init_linear_recognize
from svae.forward_models import linear_decode, linear_loglike, init_linear_loglike

from svae.models.gmm_svae import run_inference, make_gmm_global_natparam
from svae.lds import niw
from svae.hmm import dirichlet


normalize = lambda x: x / np.sum(x)

def encode(data, psi):
    J, h, _ = linear_recognize(data, psi)
    return h / (-2*J)

def decode(z, phi):
    mu, log_sigmasq = linear_decode(z, phi)
    return mu.mean(1)

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

    latent_locations = encode(data, psi)
    reconstruction = decode(latent_locations, phi)

    # make data-space plot

    # plot_or_update(0, ax0, reconstruction[:,0], reconstruction[:,1],
    #                color='b', marker='x', linestyle='', alpha=0.)

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        x, y = generate_ellipse(mu, Sigma)
        transformed_x, transformed_y = decode(np.vstack((x, y)).T, phi).T
        plot_or_update(idx, ax0, transformed_x, transformed_y,
                       alpha=min(1., K*weight), linestyle='-', linewidth=2)

    # make latent space plot

    plot_or_update(0, ax1, latent_locations[:,0], latent_locations[:,1],
                   color='k', marker='o', linestyle='')

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        x, y = generate_ellipse(mu, Sigma)
        plot_or_update(idx+1, ax1, x, y, alpha=min(1., K*weight),
                       linestyle='-', linewidth=2)

    ax1.relim()
    ax1.autoscale_view(True, True, True)

    # save plot

    plt.savefig('figures/gmm_{:04d}.png'.format(itr))
    plt.pause(0.0001)


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
    features[:,0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 10*npr.permutation(np.einsum('ti,tij->tj', features, rotations))


if __name__ == "__main__":
    npr.seed(1)
    # plt.ion()

    K = 8  # number of components in mixture model
    N = 2  # number of latent dimensions
    P = 2  # number of observation dimensions

    ## generate synthetic data
    # data = make_gmm_data()
    data = make_pinwheel_data(0.3, 0.1, 5, 100, 0.3)

    # set prior natparam
    prior_natparam = make_gmm_global_natparam(K, N, alpha=0.1/K, niw_conc=2., random=True)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, linear_recognize, linear_loglike, prior_natparam)

    # set up plotting and callback
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(data[:,0], data[:,1], s=50, color='k', marker='+')
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].autoscale(False)
    fig.tight_layout()

    itr = 0
    def callback(_, vals, natgrad, params):
        global itr
        itr += 1
        print('{}: {}'.format(itr, np.mean(vals)))
        plot(itr, axs, data, params)

    ## instantiate optimizer
    optimize = adam(data, gradfun, callback)

    ## set initialization to something generic
    # init_phi, init_psi = init_linear_loglike(N, P), init_linear_recognize(N, P)
    init_phi = init_psi = np.eye(P), 0.01*np.eye(P)
    params = prior_natparam, init_phi, init_psi

    ## optimize
    plot(0, axs, data, params)  # initial condition
    params = optimize(params, 10., 1e-3, num_epochs=600, seq_len=25, num_samples=10)
