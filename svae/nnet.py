from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import make_tuple
from toolz import curry
from collections import defaultdict

from util import compose, sigmoid, relu, identity, log1pexp, isarray


### util

def rand_partial_isometry(m, n):
    d = max(m, n)
    return np.linalg.qr(npr.randn(d,d))[0][:m,:n]

def _make_ravelers(input_shape):
    ravel = lambda inputs: np.reshape(inputs, (-1, input_shape[-1]))
    unravel = lambda outputs: np.reshape(outputs, input_shape[:-1] + (-1,))
    return ravel, unravel


### basic layer stuff

layer = curry(lambda nonlin, W, b, inputs: nonlin(np.dot(inputs, W) + b))
init_layer_random = curry(lambda d_in, d_out, scale:
                          (scale*npr.randn(d_in, d_out), scale*npr.randn(d_out)))
init_layer_partial_isometry = lambda d_in, d_out: \
    (rand_partial_isometry(d_in, d_out), npr.randn(d_out))
init_layer = lambda d_in, d_out, fn=init_layer_random(scale=1e-2): fn(d_in, d_out)


### special output layers to produce Gaussian parameters

@curry
def gaussian_mean(inputs, sigmoid_mean=False):
    mu_input, sigmasq_input = np.split(inputs, 2, axis=-1)
    mu = sigmoid(mu_input) if sigmoid_mean else mu_input
    sigmasq = log1pexp(sigmasq_input)
    return make_tuple(mu, sigmasq)

@curry
def gaussian_info(inputs):
    J_input, h = np.split(inputs, 2, axis=-1)
    J = -1./2 * log1pexp(J_input)
    return make_tuple(J, h)


### multi-layer perceptrons (MLPs)

@curry
def _mlp(nonlinearities, params, inputs):
    ravel, unravel = _make_ravelers(inputs.shape)
    eval_mlp = compose(layer(nonlin, W, b)
                       for nonlin, (W, b) in zip(nonlinearities, params))
    out = eval_mlp(ravel(inputs))
    return unravel(out) if isarray(out) else map(unravel, out)

def init_mlp(d_in, layer_specs, **kwargs):
    dims = [d_in] + [l[0] for l in layer_specs]
    nonlinearities = [l[1] for l in layer_specs]
    params = [init_layer(d_in, d_out, *spec[2:])
              for d_in, d_out, spec in zip(dims[:-1], dims[1:], layer_specs)]
    return _mlp(nonlinearities), params


### turn a gaussian_mean MLP into a log likelihood function

def _diagonal_gaussian_loglike(x, mu, sigmasq):
    mu = mu if mu.ndim == 3 else mu[:,None,:]
    T, K, p = mu.shape
    assert x.shape == (T, p)
    return -T*p/2.*np.log(2*np.pi) + (-1./2*np.sum((x[:,None,:]-mu)**2 / sigmasq)
                                      -1./2*np.sum(np.log(sigmasq))) / K

def make_loglike(gaussian_mlp):
    def loglike(params, inputs, targets):
        return _diagonal_gaussian_loglike(targets, *gaussian_mlp(params, inputs))
    return loglike


### our version of Gaussian resnets

gaussian_mlp_types = {gaussian_mean.func: 'mean', gaussian_info.func: 'info'}

def gaussian_mlp_type(layer_specs):
    return gaussian_mlp_types[layer_specs[-1][1].func]

@curry
def _gresnet(mlp_type, mlp, params, inputs):
    ravel, unravel = _make_ravelers(inputs.shape)
    mlp_params, (W, b1, b2) = params

    if mlp_type == 'mean':
        mu_mlp, sigmasq_mlp = mlp(mlp_params, inputs)
        mu_res = unravel(np.dot(ravel(inputs), W) + b1)
        sigmasq_res = log1pexp(b2)
        return make_tuple(mu_mlp + mu_res, sigmasq_mlp + sigmasq_res)
    else:
        J_mlp, h_mlp = mlp(mlp_params, inputs)
        J_res = log1pexp(b2)
        h_res = unravel(np.dot(ravel(inputs), W) + b1)
        return make_tuple(J_mlp + J_res, h_mlp + h_res)

def init_gresnet(d_in, layer_specs):
    d_out = layer_specs[-1][0] // 2
    res_params = rand_partial_isometry(d_in, d_out), np.zeros(d_out), np.zeros(d_out)
    mlp, mlp_params = init_mlp(d_in, layer_specs)
    return _gresnet(gaussian_mlp_type(layer_specs), mlp), (mlp_params, res_params)
