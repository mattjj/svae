from __future__ import division
from svae.util import softmax

expectedstats = softmax

def logZ(natparam):
    return np.sum(logsumexp(natparam, axis=-1))
