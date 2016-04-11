from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import make_tuple
from functools import partial

from nnet import tanh_layer, linear_layer, compose, init_layer
from lds.gaussian import pair_mean_to_natural

from util import sigmoid, add

# size conventions:
#   T = length of data sequence
#   n = dimension of latent state space
#   p = dimension of data
#   K = number of Monte Carlo samples
# so x.shape == (T, p) and zshape == (T, n) or (T, K, n)


### linear recognition function matching linear Gaussian SLDS parameterization

def linear_recognize(x, psi):
    C, D = psi
    J, Jzx, Jxx, _ = pair_mean_to_natural(C, np.dot(D, D.T))
    T = x.shape[0]

    J = np.tile(np.diag(J), (T, 1))
    h = np.dot(x, Jzx.T)
    logZ = np.zeros(x.shape[0])

    return make_tuple(J, h, logZ)


def init_linear_recognize(n, p, scale=1e-2):
    # return 1e-2*npr.randn(p, n), 1e-2*npr.randn(p, p)
    return scale*npr.randn(p, n), scale*npr.randn(p, p)


### mlp recognition function

def mlp_recognize(x, psi, tanh_scale=10.):
    nnet_params, ((W_h, b_h), (W_J, b_J)) = psi[:-2], psi[-2:]
    T, p = x.shape[0], b_h.shape[0]
    shape = x.shape[:-1] + (-1,)

    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    h = linear_layer(W_h, b_h)
    log_J = linear_layer(W_J, b_J)

    nnet_outputs = nnet(np.reshape(x, (-1, x.shape[-1])))
    J = -1./2 * np.exp(tanh_scale * np.tanh(log_J(nnet_outputs) / tanh_scale))
    h = h(nnet_outputs)
    logZ = np.zeros(shape[:-1])

    return make_tuple(np.reshape(J, shape), np.reshape(h, shape), logZ)


def init_mlp_recognize(hdims, n, p, scale=1e-2):
    dims = [p] + hdims
    nnet_params = map(partial(init_layer, scale=scale), zip(dims[:-1], dims[1:]))
    W_mu, b_mu = init_layer((dims[-1], n), scale)
    W_sigma, b_sigma = init_layer((dims[-1], n), scale)
    return nnet_params + [(W_mu, b_mu), (W_sigma, b_sigma)]


### residual network recognize

def resnet_recognize(x, psi):
    psi_linear, psi_mlp = psi
    # return add(linear_recognize(x, psi_linear), mlp_recognize(x, psi_mlp, tanh_scale=100.))
    return linear_recognize(x, psi_linear)


def init_resnet_recognize(hdims, n, p):
    return init_linear_recognize(n, p), init_mlp_recognize(hdims, n, p)


### meta

def sideinfo(recognize, splitter):
    def wrapped_recognize(xtilde, psi):
        x, aux = splitter(xtilde)
        nn_potentials = recognize(x, psi)
        return nn_potentials, aux
    return wrapped_recognize

mlp_recognize_withlabels = sideinfo(mlp_recognize, lambda x: (x[:,:-1], x[:,-1].astype('int')))
