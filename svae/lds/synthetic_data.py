from __future__ import division
import numpy as np
import numpy.random as npr

from svae.util import rand_psd


def rand_lds(n, p, T=None):
    def rand_stable(n):
        A = npr.randn(n, n)
        A /= np.max(np.abs(np.linalg.eigvals(A))) + 0.1
        assert np.all(np.abs(np.linalg.eigvals(A)) < 1.)
        return A

    homog = T is None

    mu_init = npr.randn(n)
    sigma_init = rand_psd(n)

    A = rand_stable(n)
    B = npr.randn(n, n)
    C = npr.randn(p, n) if homog else npr.randn(T, p, n)
    D = npr.randn(p, p) if homog else npr.randn(T, p, p)

    sigma_states = np.dot(B, B.T)
    sigma_obs = np.dot(D, D.T) if homog else np.einsum('tij,tkj->tik', D, D)

    return mu_init, sigma_init, A, sigma_states, C, sigma_obs


def generate_data(T, mu_init, sigma_init, A, sigma_states, C, sigma_obs):
    p, n = C.shape[-2:]
    states = np.empty((T, n))
    data = np.empty((T, p))

    B = np.linalg.cholesky(sigma_states)
    D = np.linalg.cholesky(sigma_obs)

    broadcast = lambda X, T: X if X.ndim == 3 else [X]*T
    As, Bs, Cs, Ds = map(broadcast, [A, B, C, D], [T-1, T-1, T, T])

    states[0] = mu_init + np.dot(np.linalg.cholesky(sigma_init), npr.randn(n))
    data[0] = np.dot(Cs[0], states[0]) + np.dot(Ds[0], npr.randn(p))
    for t, (A, B, C, D) in enumerate(zip(As, Bs, Cs[1:], Ds[1:])):
        states[t+1] = np.dot(A, states[t]) + np.dot(B, npr.randn(n))
        data[t+1] = np.dot(C, states[t+1]) + np.dot(D, npr.randn(p))

    return states, data
