from __future__ import division
import autograd.numpy as np

from recognition_models import init_mlp_recognize, init_linear_recognize
from nnet import tanh_layer, linear_layer, compose
from util import sigmoid

# size conventions:
#   T = length of data sequence
#   n = dimension of latent state space
#   p = dimension of data
#   K = number of Monte Carlo samples
# so x.shape == (T, p) and zshape == (T, n) or (T, K, n)


### dense linear gaussian forward model

def linear_loglike(x, z, phi):
    C, D = phi
    z = z if z.ndim == 3 else z[:,None,:]  # ensure z.shape == (T, K, n)
    T, p = x.shape
    K = z.shape[0]

    sigma_obs = np.dot(D, D.T)
    centered = x[:,None,:] - np.dot(z, C.T)
    quad = -1./(2*K) * np.einsum('tki,ij,tkj->', centered, np.linalg.inv(sigma_obs), centered)
    logdet = -T/2. * np.linalg.slogdet(sigma_obs)[1]
    basedensity = -T*p/2. * np.log(2*np.pi)

    return (quad + logdet + basedensity) / z.shape[1]


init_linear_loglike = init_linear_recognize


### mlp gaussian forward model

def _diagonal_gaussian_loglike(x, mu, log_sigmasq):
    T, K, p = mu.shape
    assert x.shape == (T, p)
    return -T*p/2.*np.log(2*np.pi) + (-1./2*np.sum((x[:,None,:]-mu)**2 / np.exp(log_sigmasq))
            - 1/2.*np.sum(log_sigmasq)) / K


def mlp_decode(z, phi, tanh_scale=10.):
    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = phi[:-2], phi[-2:]
    z = z if z.ndim == 3 else np.reshape(z, (-1,) + z.shape[-2:])

    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    mu = linear_layer(W_mu, b_mu)
    log_sigmasq = linear_layer(W_sigma, b_sigma)

    nnet_outputs = nnet(np.reshape(z, (-1, z.shape[-1])))
    mu = sigmoid(mu(nnet_outputs))
    log_sigmasq = tanh_scale * np.tanh(log_sigmasq(nnet_outputs) / tanh_scale)

    shape = z.shape[:-1] + (-1,)
    return mu.reshape(shape), log_sigmasq.reshape(shape)


def mlp_loglike(x, z, phi, tanh_scale=10.):
    mu, log_sigmasq = mlp_decode(z, phi, tanh_scale)
    return _diagonal_gaussian_loglike(x, mu, log_sigmasq)


def mlp_loglike_withlabels(x, z, phi):
    return mlp_loglike(x[:,:-1], z, phi)


def init_mlp_loglike(hdims, n, p):
    return init_mlp_recognize(hdims, p, n)
