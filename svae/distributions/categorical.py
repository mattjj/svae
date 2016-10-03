from __future__ import division
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from svae.util import softmax

expectedstats = softmax

def logZ(natparam):
    return np.sum(logsumexp(natparam, axis=-1))
