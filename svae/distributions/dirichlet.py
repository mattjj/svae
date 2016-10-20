from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import digamma, gammaln

def expectedstats(natparam):
    alpha = natparam + 1
    return digamma(alpha) - digamma(np.sum(alpha, -1, keepdims=True))

def logZ(natparam):
    alpha = natparam + 1
    return np.sum(np.sum(gammaln(alpha), -1) - gammaln(np.sum(alpha, -1)))
