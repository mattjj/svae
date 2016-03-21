from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal as mvn

from svae.lds.synthetic_data import generate_data, rand_lds

from lds_inference_alt import filter_forward
from test_util import bmat

npr.seed(0)


### util

def get_n(lds):
    return lds[0].shape[0]


def lds_to_big_Jh(data, lds):
    mu_init, sigma_init, A, sigma_states, C, sigma_obs = lds
    p, n = C.shape
    T = data.shape[0]

    h = C.T.dot(np.linalg.solve(sigma_obs, data.T)).T
    h[0] += np.linalg.solve(sigma_init, mu_init)

    J = np.kron(np.eye(T),C.T.dot(np.linalg.solve(sigma_obs,C)))
    J[:n,:n] += np.linalg.inv(sigma_init)
    ss_inv = np.linalg.inv(sigma_states)
    pairblock = bmat([[A.T.dot(ss_inv).dot(A), -A.T.dot(ss_inv)],
                      [-ss_inv.dot(A), ss_inv]])
    for t in range(0,n*(T-1),n):
        J[t:t+2*n,t:t+2*n] += pairblock

    return J.reshape(T*n,T*n), h.reshape(T*n)


def dense_filter(data, lds):
    n = get_n(lds)
    T = data.shape[0]

    def filtering_model(t):
        return lds_to_big_Jh(data[:t], lds)

    def dense_filtered_mu_sigma(t):
        J, h = filtering_model(t)
        mu = np.linalg.solve(J, h)
        sigma = np.linalg.inv(J)
        return mu[-n:], sigma[-n:,-n:]

    return zip(*[dense_filtered_mu_sigma(t) for t in range(1, T+1)])


def dense_loglike(data, lds):
    T = data.shape[0]
    mu_init, sigma_init, A, sigma_states, C, sigma_obs = lds

    prior_lds = mu_init, sigma_init, A, sigma_states, np.zeros_like(C), sigma_obs
    J, h = lds_to_big_Jh(data, prior_lds)

    mu_x = np.linalg.solve(J, h)
    sigma_x = np.linalg.inv(J)
    bigC = np.kron(np.eye(T), C)
    mu_y = np.dot(bigC, mu_x)
    sigma_y = np.dot(np.dot(bigC, sigma_x), bigC.T) + np.kron(np.eye(T), sigma_obs)

    return mvn.logpdf(data.ravel(), mu_y, sigma_y)


### tests

def test_filter():
    def check_filter(data, lds):
        (filtered_mus, filtered_sigmas), loglike = filter_forward(data, *lds)
        filtered_mus2, filtered_sigmas2 = dense_filter(data, lds)
        loglike2 = dense_loglike(data, lds)

        assert all(map(np.allclose, filtered_mus, filtered_mus2))
        assert all(map(np.allclose, filtered_sigmas, filtered_sigmas2))
        assert np.isclose(loglike, loglike2)

    for _ in xrange(10):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,20)
        lds = rand_lds(n, p)
        _, data = generate_data(T, *lds)

        yield check_filter, data, lds
