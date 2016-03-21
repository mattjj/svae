from __future__ import division
import autograd.numpy as np

from svae.util import solve_symmetric


def symmetrize(A):
    return (A + A.T)/2.


def is_psd(A):
    return np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) >= 0.)


def is_posdef(A):
    return np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0.)


def natural_predict(J, h, J11, J12, J22, logZ):
    # convert from natural parameter to the usual J definitions
    J, J11, J12, J22 = -2*J, -2*J11, -J12, -2*J22

    assert is_psd(J)
    assert is_psd(J11)
    assert is_psd(J22)

    Jnew = J + J11
    J_predict = symmetrize(J22 - np.dot(J12.T, np.linalg.solve(Jnew, J12)))
    h_predict = -np.dot(J12.T, solve_symmetric(Jnew, h))

    lognorm = 1./2*np.dot(h, np.linalg.solve(Jnew, h)) \
        - 1./2*np.linalg.slogdet(Jnew)[1]

    assert is_posdef(J_predict)

    return (-1./2*J_predict, h_predict), lognorm + logZ


def natural_lognorm(J, h):
    J = -2*J
    return 1./2*np.dot(h, np.linalg.solve(J, h)) \
        - 1./2*np.linalg.slogdet(J)[1]
