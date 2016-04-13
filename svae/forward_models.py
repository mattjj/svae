from __future__ import division
import autograd.numpy as np

from recognition_models import init_mlp_recognize, init_linear_recognize
from nnet import tanh_layer, linear_layer, compose
from util import sigmoid, add

# size conventions:
#   T = length of data sequence
#   n = dimension of latent state space
#   p = dimension of data
#   K = number of Monte Carlo samples
# so x.shape == (T, p) and zshape == (T, n) or (T, K, n)


def _diagonal_gaussian_loglike(x, mu, log_sigmasq):
    T, K, p = mu.shape
    assert x.shape == (T, p)
    return -T*p/2.*np.log(2*np.pi) + (-1./2*np.sum((x[:,None,:]-mu)**2 / np.exp(log_sigmasq))
            - 1/2.*np.sum(log_sigmasq)) / K


### dense linear gaussian forward model

def linear_decode(z, phi):
    C, d = phi
    z = z if z.ndim == 3 else z[:,None,:]  # ensure z.shape == (T, K, n)

    mu = np.dot(z, C.T)
    log_sigmasq = np.tile(d[None,None,:], mu.shape[:2] + (1,))

    shape = z.shape[:-1] + (-1,)
    return np.reshape(mu, shape), np.reshape(log_sigmasq, shape)


def linear_loglike(x, z, phi):
    mu, log_sigmasq = linear_decode(z, phi)
    return _diagonal_gaussian_loglike(x, mu, log_sigmasq)


init_linear_loglike = init_linear_recognize


### mlp gaussian forward model

def mlp_decode(z, phi, tanh_scale=10., sigmoid_output=True):
    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = phi[:-2], phi[-2:]
    z = z if z.ndim == 3 else z[:,None,:]  # ensure z.shape == (T, K, n)

    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    mu = linear_layer(W_mu, b_mu)
    log_sigmasq = linear_layer(W_sigma, b_sigma)

    nnet_outputs = nnet(np.reshape(z, (-1, z.shape[-1])))
    mu = sigmoid(mu(nnet_outputs)) if sigmoid_output else mu(nnet_outputs)
    log_sigmasq = tanh_scale * np.tanh(log_sigmasq(nnet_outputs) / tanh_scale)

    shape = z.shape[:-1] + (-1,)
    return mu.reshape(shape), log_sigmasq.reshape(shape)


def mlp_loglike(x, z, phi, tanh_scale=10.):
    mu, log_sigmasq = mlp_decode(z, phi, tanh_scale)
    return _diagonal_gaussian_loglike(x, mu, log_sigmasq)


def mlp_loglike_withlabels(x, z, phi):
    return mlp_loglike(x[:,:-1], z, phi)


def init_mlp_loglike(hdims, n, p, scale=1e-2):
    return init_mlp_recognize(hdims, p, n, scale)


### resnet forward model

def resnet_loglike(x, z, phi):
    mu, log_sigmasq = resnet_decode(z, phi)
    return _diagonal_gaussian_loglike(x, mu, log_sigmasq)

def resnet_decode(z, phi):
    phi_linear, phi_mlp = phi
    return add(linear_decode(z, phi_linear), mlp_decode(z, phi_mlp, tanh_scale=2., sigmoid_output=False))
    # return linear_decode(z, phi_linear)

def init_resnet_loglike(hdims, n, p):
    return init_linear_loglike(n, p), init_mlp_loglike(hdims, n, p)
