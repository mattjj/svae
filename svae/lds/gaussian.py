from __future__ import division
import autograd.numpy as np
from numpy.random.mtrand import _rand as rng

from svae.util import solve_triangular, solve_posdef_from_cholesky, symmetrize

# TODO remove these
from inspect import getcallargs, getargspec
from svae.util import contract

### distribution form

def sample(mu, sigma, rng=rng):
    L = np.linalg.cholesky(sigma)
    return mu + np.dot(L, rng.normal(size=mu.shape))


def predict(mu, sigma, A, sigma_update):
    mu_predict = np.dot(A, mu)
    sigma_predict = np.dot(A, np.dot(sigma, A.T)) + sigma_update
    return mu_predict, sigma_predict


def condition_on(mu, sigma, A, y, sigma_obs):
    temp1 = np.dot(A, sigma)
    sigma_pred = np.dot(temp1, A.T) + sigma_obs
    L = np.linalg.cholesky(sigma_pred)
    v = solve_triangular(L, y - np.dot(A, mu))
    ll = -1./2 * np.dot(v, v) - np.sum(np.log(np.diag(L))) \
        - y.shape[0]/2.*np.log(2*np.pi)
    mu_cond = mu + np.dot(temp1.T, solve_triangular(L, v, 'T'))

    temp2 = solve_triangular(L, temp1)
    sigma_cond = sigma - np.dot(temp2.T, temp2)

    return (mu_cond, sigma_cond), ll


### natural parameter form

def natural_condition_on(J, h, y, Jxx, Jxy, Jyy=None, logZ=None):
    # NOTE: assumes Jxy is *negative* definite, usually h - np.dot(Jxy, y.T).T
    J_cond, h_cond = J + Jxx, h + np.dot(Jxy, y.T).T

    if Jyy is None or logZ is None:
        return J_cond, h_cond
    return (J_cond, h_cond), logZ + np.dot(y, np.dot(Jyy, y.T))


def natural_condition_on_general(J, h, Jo, ho, logZo):
    Jo = Jo if Jo.ndim == 2 else np.diag(Jo)
    assert np.all(np.linalg.eigvals(-2*(J + Jo)) > 0)
    return (J + Jo, h + ho), logZo


def natural_predict(J, h, J11, J12, J22, logZ):
    # convert from natural parameter to the usual J definitions
    J, J11, J12, J22 = -2*J, -2*J11, -J12, -2*J22

    L = np.linalg.cholesky(J + J11)
    v = solve_triangular(L, h)
    lognorm = 1./2*np.dot(v,v) - np.sum(np.log(np.diag(L)))
    h_predict = -np.dot(J12.T, solve_triangular(L, v, trans='T'))

    temp = solve_triangular(L, J12)
    J_predict = J22 - np.dot(temp.T, temp)

    assert np.all(np.linalg.eigvals(J_predict) > 0)

    return (-1./2*J_predict, h_predict), lognorm + logZ


def natural_sample(J, h, num_samples=None, rng=rng):
    sample_shape = (num_samples,) + h.shape if num_samples else h.shape
    J = -2*J
    if J.ndim == 1:
        return h / J + rng.normal(size=sample_shape) / np.sqrt(J)
    else:
        L = np.linalg.cholesky(J)
        noise = solve_triangular(L, rng.normal(size=sample_shape).T, trans='T')
        return solve_posdef_from_cholesky(L, h.T).T + noise.T


def natural_lognorm(J, h):
    J = -2*J
    L = np.linalg.cholesky(J)
    v = solve_triangular(L, h)
    return 1./2*np.dot(v, v) - np.sum(np.log(np.diag(L)))


def natural_rts_backward_step(next_smooth, next_pred, filtered, pair_param):
    # p = "predicted", f = "filtered", s = "smoothed", n = "next"
    (Jns, hns, mun), (Jnp, hnp), (Jf, hf) = next_smooth, next_pred, filtered
    J11, J12, J22, _ = pair_param

    # convert from natural parameter to the usual J definitions
    Jns, Jnp, Jf, J11, J12, J22 = -2*Jns, -2*Jnp, -2*Jf, -2*J11, -J12, -2*J22

    J11, J12, J22 = Jf + J11, J12, Jns - Jnp + J22
    L = np.linalg.cholesky(J22)
    temp = solve_triangular(L, J12.T)
    Js = J11 - np.dot(temp.T, temp)
    hs = hf - np.dot(temp.T, solve_triangular(L, hns - hnp))

    mu, sigma = info_to_mean((Js, hs))
    ExnxT = -solve_posdef_from_cholesky(L, np.dot(J12.T, sigma)) + np.outer(mun, mu)

    ExxT = sigma + np.outer(mu, mu)

    return -1./2*Js, hs, (mu, ExxT, ExnxT)

### converting

def mean_to_natural(mu, sigma):
    neghalfJ = -1./2*np.linalg.inv(sigma)
    h = np.linalg.solve(sigma, mu)
    logZ = -1./2*np.dot(mu, h) - 1./2*np.linalg.slogdet(sigma)[1]
    return neghalfJ, h, logZ


def info_to_mean(infoparams):
    J, h = infoparams
    Sigma = np.linalg.inv(J)
    mu = np.dot(Sigma, h)
    return mu, Sigma


def natural_to_mean(natparam):
    J, h = natparam
    J = -2*J
    return info_to_mean((J, h))


def pair_mean_to_natural(A, sigma):
    assert 2 <= A.ndim == sigma.ndim <= 3
    ndim = A.ndim

    einstring = 'tji,tjk->tik' if ndim == 3 else 'ji,jk->ik'
    trans = (0, 2, 1) if ndim == 3 else (1, 0)
    temp = np.linalg.solve(sigma, A)

    Jxx = -1./2 * np.einsum(einstring, A, temp)
    Jxy = np.transpose(temp, trans)
    Jyy = -1./2 * np.linalg.inv(sigma)
    logZ = -1./2 * np.linalg.slogdet(sigma)[1]

    return Jxx, Jxy, Jyy, logZ


### vanilla gaussian stuff, maybe should be in a different file

def expectedstats(natparam):
    neghalfJ, h, _, _ = unpack_dense(natparam)
    J = -2*neghalfJ
    Ex = np.linalg.solve(J, h)
    ExxT = np.linalg.inv(J) + Ex[...,None] * Ex[...,None,:]
    En = np.ones(J.shape[0]) if J.ndim == 3 else 1.
    return pack_dense((ExxT, Ex, En, En))

def logZ(natparam):
    neghalfJ, h, a, b = unpack_dense(natparam)
    J = -2*neghalfJ
    L = np.linalg.cholesky(J)
    return 1./2 * np.sum(h * np.linalg.solve(J, h)) \
        - np.sum(np.log(np.diagonal(L, axis1=-1, axis2=-2))) \
        - np.sum(a) - np.sum(b)

### packing and unpacking node potentials into dense matrices

vs, hs = partial(np.concatenate, axis=-2), partial(np.concatenate, axis=-1)
tr = partial(np.swapaxes, axis1=-1, axis2=-2)

def pack_dense(node_potentials):
    '''Packs (J, h, a, b) tuple, where J has shape (T, N), h has shape (T, N),
    and a and b both have shape (T, 1), into a dense ndarray of shape
    (T, N+2, N+2), so we can use tensordot. If argument is of the form (J, h),
    then a and b are set to zero.'''
    if len(node_potentials) not in {2, 4}: raise ValueError

    Jdiag, h = node_potentials[:2]
    J = Jdiag[...,None] * np.eye(N)[None,...]
    h = 1./2 * h[...,None]
    T, N = h.shape
    z1, z2 = np.zeros((T, N, 1)), np.zeros((T, 1, 1))
    a, b = node_potentials[2:] if len(node_potentials) == 4 else (z2, z2)

    return vs(( hs(( J,      h,  z1 )),
                hs(( tr(h),  a,  z2 )),
                hs(( tr(z1), z2, b  ))))

def unpack_dense(arr):
    '''Unpacks dense ndarray of shape (T, N+2, N+2) into four parts, with shapes
      (T, N, N), (T, N), (T,), and (T,).'''
    T, N = arr.shape[0], arr.shape[1] - 2
    return arr[:,:N, :N], arr[:,N,:N], arr[:,N,N], arr[:,N+1,N+1]
