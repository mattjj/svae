import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from svae.lds.cython_lds_inference import _compute_stats, _compute_stats_grad
from svae.util import randn_like, allclose, contract, make_unop, shape

T, N = 4, 2

Ex    = np.require(npr.randn(N, T)   , np.double, 'F')
ExxT  = np.require(npr.randn(N, N, T), np.double, 'F')
ExnxT = np.require(npr.randn(N, N, T-1), np.double, 'F')

def compute_stats(Ex, ExxT, ExnxT, inhomog):
    T = Ex.shape[-1]
    E_init_stats = ExxT[:,:,0], Ex[:,0], 1., 1.
    E_pair_stats = np.transpose(ExxT, (2, 0, 1))[:-1], \
        ExnxT.T, np.transpose(ExxT, (2, 0, 1))[1:], np.ones(T-1)
    E_node_stats = np.diagonal(ExxT.T, axis1=-1, axis2=-2), Ex.T, np.ones(T)

    if not inhomog:
        E_pair_stats = map(lambda x: np.sum(x, axis=0), E_pair_stats)

    return E_init_stats, E_pair_stats, E_node_stats

def test_compute_stats():
    assert allclose(compute_stats(Ex, ExxT, ExnxT, False), _compute_stats(Ex, ExxT, ExnxT, False))
    assert allclose(compute_stats(Ex, ExxT, ExnxT, True), _compute_stats(Ex, ExxT, ExnxT, True))


def test_compute_stats_grad():
    F = make_unop(lambda x: np.require(x, np.double, 'F'), tuple)

    dotter = F(randn_like(compute_stats(Ex, ExxT, ExnxT, True)))
    g1 = grad(lambda x: contract(dotter, compute_stats(*x)))((Ex, ExxT, ExnxT, 1.))
    g2 = _compute_stats_grad(dotter)
    assert allclose(g1[:3], g2)

    dotter = F(randn_like(compute_stats(Ex, ExxT, ExnxT, False)))
    g1 = grad(lambda x: contract(dotter, compute_stats(*x)))((Ex, ExxT, ExnxT, 0.))
    g2 = _compute_stats_grad(dotter)
    assert allclose(g1[:3], g2)
