from __future__ import division
import tensorflow as tf
sigmoid, relu, tanh = tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh
from toolz import curry

from util import compose, identity


### util

log1p = lambda x: tf.log(1. + tf.exp(x))
identity = lambda x: x


### basic layer stuff

layer = curry(lambda nonlin, W, b, inputs: nonlin(tf.matmul(inputs, W) + b))
_randn = lambda shape, scale: tf.Variable(tf.random_normal(shape, stddev=scale))
init_layer_random = curry(lambda m, n, scale:
                          (_randn((m, n), scale), _randn((n,), scale)))
init_layer = lambda m, n, fn=init_layer_random(scale=1e-2): fn(m, n)


### MLPs

@curry
def _mlp(nonlinearities, params, inputs):
    eval_mlp = compose(layer(nonlin, W, b)
                       for nonlin, (W, b) in zip(nonlinearities, params))
    return eval_mlp(inputs)

def init_mlp(m, layer_specs):
    dims = [m] + [l[0] for l in layer_specs]
    nonlinearities = [l[1] for l in layer_specs]
    params = [init_layer(m, n, *spec[2:])
              for m, n, spec in zip(dims[:-1], dims[1:], layer_specs)]
    return _mlp(nonlinearities, params), params

### special output layers to produce Gaussian parameters

@curry
def gaussian_mean(inputs, sigmoid_mean=False):
    mu_input, sigmasq_input = tf.split(1, 2, inputs)
    mu = sigmoid(mu_input) if sigmoid_mean else mu_input
    sigmasq = log1p(tf.exp(sigmasq_input))
    return mu, sigmasq

### turn a gaussian_mean MLP into a log likelihood function

def _diagonal_gaussian_loglike(x, mu, sigmasq):
  mu_shape = tf.shape(mu, out_type=tf.float32)
  T, K, p = mu_shape[0], mu_shape[1], mu_shape[1]
  return -T*p/2.*tf.log(2.*np.pi) \
      + (-1./2*tf.sum((tf.expand_dims(x, 1) - mu)**2 / sigmasq)
         -1./2*tf.sum(tf.log(sigmasq))) / K

def make_loglike(gaussian_mlp):
    def loglike(params, inputs, targets):
        return _diagonal_gaussian_loglike(targets, *gaussian_mlp(params, inputs))
    return loglike
