from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr


### gaussian inference functions

# the prior and variational distribution are factored across datapoints.
# the inference functions here operate on batches of datapoints, so they
# perform inference operations on batches of independent diagonal gaussians.

# the code here is specialized to diagonal gaussians

def unpack_params(natparams):
    neghalfJs, hs, _ = natparams
    Js = -2 * neghalfJs
    mus, sigmasqs = hs/Js, 1./Js
    return mus, sigmasqs


def sample(natparams, num_samples=None):
    'Takes a list of natural parameter tuples and produces an array of samples'
    mus, sigmasqs = unpack_params(natparams)
    T, n = mus.shape
    rand_shape = (T, num_samples, n) if num_samples else (T, n)
    return mus[:,None,:] + np.sqrt(sigmasqs)[:,None,:] * npr.normal(size=rand_shape)


def gaussian_vlb(variational_natparams):
    'Takes a list of variational natural parameters output by the recognition network'
    'and returns E_q[log p(x)/q(x)] = -kl( q(x) || p(x) ) where p(x) is N(0,I)'
    mus, sigmasqs = unpack_params(variational_natparams)
    return 1./2 * np.sum(1 + np.log(sigmasqs) - mus**2 - sigmasqs)


### inference function

# prior_natparam and global_natparam are ignored because there are no global
# variables in this model and hence no variational factors on global variables.
# correspondingly, there's no expected stats to return.

def run_inference(prior_natparam, global_natparam, nn_potential, num_samples):
    samples = sample(nn_potential, num_samples)
    local_vlb = gaussian_vlb(nn_potential)
    global_vlb = 0.
    return samples, (), global_vlb, local_vlb
