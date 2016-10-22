from __future__ import division, print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam, sgd
from svae.svae import make_gradfun
from svae.nnet import init_gresnet, make_loglike, gaussian_mean, gaussian_info
from svae.models.gmm import pgm_functions

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
    plt.ion()

    num_clusters = 5           # number of clusters in pinwheel data
    samples_per_cluster = 100  # number of samples per cluster in pinwheel
    K = 15                     # number of components in mixture model
    N = 2                      # number of latent dimensions
    P = 2                      # number of observation dimensions

    # generate synthetic data
    data = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

    # set up model
    init_pgm_param, run_inference, make_encoder_decoder, unflatten = \
        pgm_functions(K, N)

    # set prior natparam to something sparsifying but otherwise generic
    pgm_prior_params = init_pgm_param(alpha=0.1/K, niw_conc=0.5, random_scale=0.)

    # construct recognition and decoder networks and initialize them
    recognize, recogn_params = \
        init_gresnet(P, [(40, np.tanh), (40, np.tanh), (2*N, gaussian_info)])
    decode,   loglike_params = \
        init_gresnet(N, [(40, np.tanh), (40, np.tanh), (2*P, gaussian_mean)])
    loglike = make_loglike(decode)

    # initialize gmm parameters
    pgm_params = init_pgm_param(alpha=1., niw_conc=1., random_scale=3.)
    params = pgm_params, loglike_params, recogn_params

    # set up encoder/decoder and plotting
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)

    # instantiate svae gradient function
    gradfun = make_gradfun(run_inference, recognize, loglike, pgm_prior_params, data)

    # optimize
    params = sgd(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e3),
                 params, num_iters=1000, step_size=1e-2)
