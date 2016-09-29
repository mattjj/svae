from __future__ import division, print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam
from svae.svae import make_gradfun
from svae.nnet import init_gresnet, make_loglike, gaussian_mean, gaussian_info
from svae.models.gmm import run_inference, init_pgm_param, make_encoder_decoder

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

    K = 15  # number of components in mixture model
    N = 2   # number of latent dimensions
    P = 2   # number of observation dimensions

    # generate synthetic data
    data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)

    # set prior natparam to something sparsifying but otherwise generic
    pgm_prior_params = init_pgm_param(K, N, alpha=0.1/K, niw_conc=0.5)

    # construct recognition and decoder networks and initialize them
    recognize, recogn_params = init_gresnet(P, [(3, np.tanh), (2*N, gaussian_info)])
    decode,   loglike_params = init_gresnet(N, [(3, np.tanh), (2*P, gaussian_mean)])
    loglike = make_loglike(decode)
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)

    # initialize gmm parameters
    pgm_params = init_pgm_param(K, N, alpha=1., niw_conc=2., random_scale=5.)
    params = pgm_params, loglike_params, recogn_params

    # instantiate svae gradient function
    gradfun = make_gradfun(run_inference, recognize, loglike, pgm_prior_params, data)

    # optimize
    params = adam(gradfun(batch_size=50, num_samples=1),
                  params, num_iters=1000, step_size=1e-3)
