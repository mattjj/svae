from __future__ import division
from autograd.scipy.special import digamma, gammaln

def expectedstats(natparam):
    alpha = natparam + 1
    return digamma(alpha) - digamma(alpha.sum(-1, keepdims=True))


def logZ(natparam):
    alpha = natparam + 1
    return gammaln(alpha).sum() - gammaln(alpha.sum())
