from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from time import time

from svae.svae import make_gradfun
from svae.optimizers import adam, adadelta
from svae.recognition_models import mlp_recognize, init_mlp_recognize
from svae.forward_models import mlp_loglike, mlp_decode, init_mlp_loglike
import svae.lds.mniw as mniw
import svae.lds.niw as niw
from svae.lds.lds_inference import natural_lds_sample

from svae.models.lds_svae import make_prior_natparam, lds_prior_expectedstats
from svae.models.lds_svae import run_inference
from svae.util import zeros_like, make_unop

from svae.models.lds_svae import cython_run_inference as run_inference
from svae.lds.lds_inference import cython_natural_lds_sample as natural_lds_sample
zero_after_prefix = lambda prefix: make_unop(lambda x: np.concatenate(
    (x[:prefix], np.zeros_like(x[prefix:]))), tuple)

from make_dot_data import make_dot_data


def generate_samples(params, data, prefix):
    _, phi, _ = params
    x = sample_conditional_states(params, data, zero_after=prefix)
    return mlp_decode(x, phi)[0]

def sample_conditional_states(params, data, zero_after=-1):
    zero = zero_after_prefix(zero_after)
    natparam, phi, psi = params
    node_potentials = zero(mlp_recognize(data, psi))
    local_natparam = lds_prior_expectedstats(natparam)
    x = natural_lds_sample(local_natparam, node_potentials, num_samples=50)
    return x

fig, ax = plt.subplots(figsize=(10, 10))
# plt.ion()
plt.tight_layout()
def plot(itr, params, data, prefix):
    y = generate_samples(params, data, prefix)
    T, num_samples, ndim = y.shape

    mean_image = y.mean(1)
    sample_images = np.hstack([y[:,i,:] for i in npr.choice(num_samples, 5, replace=False)])
    big_image = np.hstack((data, mean_image, sample_images))

    ax.matshow(big_image, cmap='gray')
    ax.autoscale(False)
    ax.axis('off')
    ax.plot([-0.5, big_image.shape[1]], [prefix-0.5, prefix-0.5], 'r', linewidth=2)

    fig.tight_layout()
    fig.savefig('dots_{:03d}.png'.format(itr))
    plt.close('all')

    # plt.draw()
    # plt.pause(0.1)


if __name__ == "__main__":
    npr.seed(0)
    np.set_printoptions(precision=2)

    # latent space dimension
    n = 6

    # set up prior
    prior_natparam = make_prior_natparam(n)

    # load pendulum data
    data = make_dot_data(20, 500, 5000, v=0.75, render_sigma=0.15, noise_sigma=0.1)
    T, p = data.shape

    # randomly initialize nnet parameters
    init_phi = init_mlp_loglike([50], n, p)    # likelihood params
    init_psi = init_mlp_recognize([50], n, p)  # recognition params

    # build svae gradient function
    gradfun = make_gradfun(run_inference, mlp_recognize, mlp_loglike, prior_natparam)

    # set up optimizers
    eq_mod = lambda a, b, m: a % m == b % m
    plot_every = 10
    prefix = 25

    def print_eigvals(params):
        natparam, phi, psi = params
        A, Sigma = mniw.expected_standard_params(natparam[1])
        print(np.array(sorted(np.abs(np.linalg.eigvals(A)), reverse=True)))

    total = lambda: None
    def callback(itr, vals, natgrad, params):
        total.params = params
        if eq_mod(itr, -1, plot_every): plot(itr, params, data[:200], prefix)
        print_eigvals(params)
        print('{} at {} sec: {}'.format(itr, time() - total.time, np.mean(vals)))

    adam_optimize = adam(data, gradfun, callback)
    adad_optimize = adadelta(data, gradfun, callback)

    # initialize params to prior on dynamics, vae fit on phi and psi
    params = prior_natparam, init_phi, init_psi

    # optimize
    total.time = time()
    params = adam_optimize(params, 1., 1e-3, num_epochs=1500, seq_len=50, num_samples=1)
