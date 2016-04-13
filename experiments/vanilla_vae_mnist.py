from __future__ import division, print_function
import numpy as np
import numpy.random as npr

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import mlp_recognize, init_mlp_recognize
from svae.forward_models import mlp_loglike, mlp_decode, init_mlp_loglike
from svae.viz import plot_random_examples, plot_samples

from svae.models.vanilla_vae import run_inference

from data import mnist

np.seterr(invalid='raise', over='raise', divide='raise')


def load_mnist(num, class_label, noise_scale):
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], -1))
    normalize = lambda x: x / x.max(1, keepdims=True)
    random_subset = lambda x, n: x[np.random.choice(x.shape[0], n)]
    add_noise = lambda x: x + noise_scale*npr.normal(size=x.shape)

    train_images, train_labels, test_images, test_labels = mnist()
    train_images = random_subset(train_images[train_labels == class_label], num)

    return normalize(partial_flatten(train_images))


if __name__ == "__main__":
    npr.seed(0)

    # load mnist data
    data = load_mnist(num=2000, class_label=5, noise_scale=1e-3)
    plot_random_examples(data)
    T, p = data.shape


    # set model size parameters
    n = 2
    recognition_model_layers = [50]
    forward_model_layers = [50]

    # no global variables, so the prior natparam on global variables is empty
    prior_natparam = ()

    # build svae gradient function
    num_minibatches = 100
    gradfun = make_gradfun(run_inference, mlp_recognize, mlp_loglike, prior_natparam)

    # set up callback function for printing and plotting during optimization
    total = lambda: None
    total.itr = 0
    def callback(itr, val, natgrad, params):
        _, phi, _ = params
        total.itr += 1

        samplefun = lambda num: mlp_decode(npr.randn(num, n), phi)[0].mean(1)
        eq_mod = lambda a, b, m: a % m == b % m

        print('{}: {}'.format(total.itr, val))
        if eq_mod(total.itr, -1, 1): plot_samples(total.itr, samplefun)

    # set up optimizer
    optimize = adam(data, gradfun, callback)

    # initialize model parameters
    init_eta = ()  # no global variables
    init_phi = init_mlp_loglike(forward_model_layers, n, p)
    init_psi = init_mlp_recognize(recognition_model_layers, n, p)
    params = init_eta, init_phi, init_psi

    # optimize
    params = optimize(params, 0., 1e-3, num_epochs=1000, seq_len=20, num_samples=1)
