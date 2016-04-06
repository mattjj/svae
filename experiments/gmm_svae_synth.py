from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import linear_recognize, init_linear_recognize
from svae.forward_models import linear_loglike, init_linear_loglike

from svae.models.gmm_svae import run_inference, make_gmm_global_natparam
from svae.lds import niw
from svae.hmm import dirichlet


def plot(ax, params):
    def plot_gaussian(ax, idx, mu, Sigma, alpha):
        t = np.hstack([np.arange(0,2*np.pi,0.01),0])
        circle = np.vstack([np.sin(t),np.cos(t)])
        ellipse = 2. * np.dot(np.linalg.cholesky(Sigma), circle)
        x, y = ellipse[0] + mu[0], ellipse[1] + mu[1]

        if len(ax.lines) > idx:
            ax.lines[idx].set_data((x, y))
            ax.lines[idx].set_alpha(alpha)
        else:
            ax.plot(x, y, linestyle='-', alpha=alpha)

    normalize = lambda x: x / np.sum(x)

    natparam, phi, psi = params
    dir_hypers, all_niw_hypers = natparam
    weights = normalize(np.exp(dirichlet.expectedstats(dir_hypers)))
    components = map(niw.expected_standard_params, all_niw_hypers)

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        plot_gaussian(ax, idx, mu, Sigma, alpha=weight)


if __name__ == "__main__":
    npr.seed(0)
    np.set_printoptions(precision=2)
    plt.ion()

    K = 5  # number of components in mixture model
    N = 2  # number of latent dimensions
    P = 2  # number of observation dimensions

    # generate synthetic data
    data = np.concatenate([-2. + npr.randn(25, P), 2. + npr.randn(25, P)])

    # set prior natparam
    prior_natparam = make_gmm_global_natparam(K, N, alpha=1., random=True)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, linear_recognize, linear_loglike, prior_natparam)

    # set up plotting and callback
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1], 1, color='k')
    ax.axis('square')
    ax.autoscale(False)

    def callback(itr, vals, natgrad, params):
        print('{}: {}'.format(itr, np.mean(vals)))
        plot(ax, params)
        plt.draw()
        plt.pause(0.01)

    # instantiate optimizer
    optimize = adam(data, gradfun, callback)

    # set initialization to something generic
    # init_phi, init_psi = init_linear_loglike(N, P), init_linear_recognize(N, P)
    init_phi = init_psi = np.eye(P), 0.01*np.eye(P)
    params = prior_natparam, init_phi, init_psi

    # optimize
    params = optimize(params, 10., 0., num_epochs=500, seq_len=len(data))
