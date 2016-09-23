from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import make_tuple
from toolz import curry
from collections import defaultdict

from util import compose, sigmoid


### layers

@curry
def layer(nonlin, W, b, inputs):
    return nonlin(np.dot(inputs, W) + b)

@curry
def init_layer_random(d_in, d_out, scale):
    return scale*npr.randn(d_in, d_out), scale*npr.randn(d_out)

def init_layer_partial_isometry(d_in, d_out):
    d = max(d_in, d_out)
    return np.linalg.qr(npr.randn(d, d))[0][:d_in,:d_out], npr.randn(d_out)

identity = lambda x: x


### multi-layer perceptrons (MLPs)

def init_mlp(d_in, layer_specs):
    dims = [d_in] + [l[0] for l in layer_specs]
    return [init_layer(d_in, d_out, *spec[2:])
            for d_in, d_out, spec in zip(dims[:-1], dims[1:], layer_specs)]

@curry
def mlp(layer_specs, params, inputs):
    nonlins = [l[1] for l in layer_specs]
    eval_mlp = compose(layer(nonlin, W, b)
                       for nonlin, (W, b) in zip(nonlins, params))
    out = eval_mlp(np.reshape(inputs, (-1, inputs.shape[-1])))
    unravel = lambda out: np.reshape(out, inputs.shape[:-1] + (-1,))
    return unravel(out) if not isinstance(out, tuple) else map(unravel, out)


### special output layers to output Gaussian parameters

def _split_inputs(inputs):
    dim = inputs.shape[-1] // 2
    return inputs[...,:dim], inputs[...,dim:]

@curry
def gaussian_mean(inputs, tanh_scale=False, sigmoid_mean=True):
    mu_input, log_sigmasq_input = _split_inputs(inputs)
    mu = sigmoid(mu_input) if sigmoid_mean else mu_input
    log_sigmasq = tanh_scale * np.tanh(log_sigmasq_input / tanh_scale) \
        if tanh_scale else log_sigmasq_input
    return mu, log_sigmasq

@curry
def gaussian_info(inputs, tanh_scale=False):
    J_input, h = _split_inputs(inputs)
    J = -1./2 * np.exp(tanh_scale * np.tanh(J_input / tanh_scale)) \
        if tanh_scale else -1./2 * np.exp(J_input)
    return make_tuple(J, h)


### turn a gaussian_mean MLP into a log likelihood function

def _diagonal_gaussian_loglike(x, mu, log_sigmasq):
    T, K, p = mu.shape
    assert x.shape == (T, p)
    return -T*p/2.*np.log(2*np.pi) + (-1./2*np.sum((x[:,None,:]-mu)**2 / np.exp(log_sigmasq))
            - 1/2.*np.sum(log_sigmasq)) / K

def make_loglike(gaussian_mlp):
    def loglike(targets, inputs, params):
        mu, log_sigmasq = gaussian_mlp(params, inputs)
        return _diagonal_gaussian_loglike(targets, mu, log_sigmasq)
    return loglike


### our version of Gaussian resnets
# TODO this is broken below here

gaussian_mlp_types = {gaussian_mean.func: 'mean', gaussian_info.func: 'info'}

def gaussian_mlp_type(layer_specs):
    return gaussian_mlp_types.get(layer_specs[-1][1].func, 'none')

def init_resnet(d_in, layer_specs):
    if gaussian_mlp_type(layer_specs) == 'none':
        d_out = layer_specs[-1][0]
        randn_partial_isometry(d_in, d_out), np.zeros(d_out)
    else:
        d_out = layer_specs[-1][0] // 2
        return rand_partial_isometry(d_in, d_out), np.zeros(d_out), np.zeros(d_out)

@curry
def gresnet(layer_specs, params, inputs):
    assert is_gaussian_mlp(layer_specs)
    ravel = lambda x: np.reshape(x, (-1, inputs.shape[-1]))
    unravel = lambda x: np.reshape(x, inputs.shape[:-1] + (-1,))

    mlp_params, (W_mu, b_mu, log_sigmasq_res) = params
    mu_mlp, log_sigmasq_mlp = mlp(layer_specs, mlp_params, inputs)
    mu_res = unravel(np.dot(ravel(inputs), W_mu) + b_mu)

    return mu_mlp + mu_res, log_sigmasq_mlp + log_sigmasq_res

# maybe we can just switch on which type it is
