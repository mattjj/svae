from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import make_tuple
from toolz import curry

from util import compose, sigmoid


### layers

@curry
def layer(nonlin, W, b, inputs):
    return nonlin(np.dot(inputs, W) + b)

@curry
def init_layer_random(d_in, d_out, scale):
    return scale*npr.randn(d_in, d_out), scale*npr.randn(d_out)

def init_layer(d_in, d_out, init_fn=init_layer_random(scale=1e-2)):
    return init_fn(d_in, d_out)

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
    reshape = lambda out: np.reshape(out, inputs.shape[:-1] + (-1,))
    return reshape(out) if not isinstance(out, tuple) else map(reshape, out)


### MLPs that output Gaussian parameters

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
