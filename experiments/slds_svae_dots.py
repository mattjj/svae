from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom

from svae.svae import make_gradfun
from svae.optimizers import adam, adadelta, sga
from svae.recognition_models import mlp_recognize, mlp_recognize_withlabels, \
    init_mlp_recognize
from svae.forward_models import mlp_loglike, mlp_loglike_withlabels, mlp_decode, init_mlp_loglike


from svae.models.slds_svae import run_inference, run_inference_withlabels, \
    make_slds_global_natparam, optimize_local_meanfield, unbox
from svae.util import zeros_like, randn_like, scale, add


use_cython = True

if use_cython:
    from svae.recognition_models import mlp_recognize_dense as mlp_recognize, \
        mlp_recognize_dense_withlabels as mlp_recognize_withlabels


def make_dot_data(k, n, speeds):
    def make_template(speed):
        scale = 100
        bigblock = gaussian_filter(np.eye(k*scale), (1.5*scale/speed, 1.5*scale*speed))
        block = zoom(bigblock, (1./(scale*speed), 1./scale))
        block /= block.max(1)[:,None]
        template = np.tile(np.vstack((block, block[1:-1,::-1])), (2, 1))
        return template

    templates = map(make_template, speeds)
    labels = npr.randint(len(speeds), size=n)
    data_blocks = [templates[label] for label in labels]
    labels = np.repeat(labels, [len(block) for block in data_blocks])
    data = np.vstack(data_blocks)
    data += 5e-2 * npr.randn(*data.shape)

    return data, labels

def decode_states(params, data):
    natparam, phi, psi = params
    hmm_global_natparam, lds_global_natparam = natparam

    node_potentials = mlp_recognize(data, psi)
    (hmm_stats, _), _, _ = optimize_local_meanfield(natparam, unbox(node_potentials))
    _, _, expected_states = hmm_stats

    return expected_states

def plot(params, data):
    tic = time()
    data = data[:500]
    states = decode_states(params, data)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.matshow(data.T, aspect='auto')
    ax2.matshow(np.eye(labels.max()+1)[labels].T, aspect='auto')  # global variable
    ax3.matshow(states.T, aspect='auto')
    plt.savefig('superhero.png')
    plt.close()

    print('plotting took {} seconds'.format(time() - tic))

if __name__ == "__main__":
    npr.seed(0)

    # latent space dimension
    n = 5

    # set up prior
    prior_natparam = make_slds_global_natparam(2, n, sticky_bias=0., random=False)

    # load pendulum data
    data, labels = make_dot_data(15, 50, speeds=[1., 1.9])
    # data, labels = make_dot_data(10, 50, speeds=[1.7])
    T, p = data.shape
    labeled_data = np.hstack((data, labels[:,None]))
    oneclass_data = np.hstack((data, np.zeros((data.shape[0], 1))))

    # plot data
    plt.matshow(data[:200].T)
    plt.set_cmap('bone')
    plt.savefig('dots2.png')
    plt.close()

    # randomly initialize nnet parameters
    init_phi = init_mlp_loglike([20], n, p)    # likelihood params
    init_psi = init_mlp_recognize([20], n, p)  # recognition params

    # build svae gradient function
    gradfun = make_gradfun(
        run_inference_withlabels, mlp_recognize_withlabels, mlp_loglike_withlabels, prior_natparam)

    # set up callback
    eq_mod = lambda a, b, m: a % m == b % m
    plot_every = 10
    def callback(itr, vals, natgrad, params):
        if eq_mod(itr, -1, plot_every): plot(params, data)
        print('{}: {}'.format(itr, np.median(vals[-10:])))

    # set up optimizers
    oneclass_adad_optimize = adadelta(oneclass_data, gradfun, callback)
    adad_optimize = adadelta(labeled_data, gradfun, callback)
    adam_optimize = adam(labeled_data, gradfun, callback)
    sga_optimize = sga(labeled_data, gradfun, callback)

    # initialize params to prior on dynamics, vae fit on phi and psi
    params = prior_natparam, init_phi, init_psi

    # optimize
    print('optimizing!')
    params = oneclass_adad_optimize(params, 1e-1, num_epochs=30, num_minibatches=100, num_samples=1)
    params = oneclass_adad_optimize(params, 1e-2, num_epochs=30, num_minibatches=50, num_samples=1)

    # with open('fit_slds_dots.pkl', 'r') as infile:
    #     params = pickle.load(infile)

    def reinitialize_lds_params(params):
        natparam, phi, psi = params
        hmm_natparam, lds_natparams = natparam
        fit_lds_natparam = lds_natparams[0]
        lds_natparams = tuple(add(scale(1e-2, randn_like(fit_lds_natparam)), fit_lds_natparam)
                              for _ in lds_natparams)
        return (hmm_natparam, lds_natparams), phi, psi

    print('fitting two states!')
    params2 = reinitialize_lds_params(params)
    params2 = adad_optimize(params2, 1e-2, num_epochs=30, num_minibatches=25, num_samples=5)

    # params = adad_optimize(params, 1e-2, num_epochs=10, num_minibatches=100, num_samples=1)
    # params = adad_optimize(params, 1e-2, num_epochs=50, num_minibatches=50,  num_samples=5)
    # params = adam_optimize(params, 1e-2, 1e-4, 25, 5)
