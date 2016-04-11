from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr

from util import sigmoid, compose


def make_layer(activation):
    def layer(W, b):
        def apply(h):
            return activation(np.dot(h, W) + b)
        return apply
    return layer

tanh_layer = make_layer(np.tanh)
sigmoid_layer = make_layer(sigmoid)
linear_layer = make_layer(lambda x: x)

def init_layer(shape, scale=1e-2):
    m, n = shape
    return scale*npr.randn(m, n), scale*npr.randn(n)
