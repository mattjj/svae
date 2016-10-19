from __future__ import division
import numpy as np
import numpy.random as npr

def rand_psd(n):
    temp = npr.randn(n,n)
    return np.dot(temp, temp.T)
