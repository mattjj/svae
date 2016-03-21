from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from pylds.lds_messages_interface import E_step as _E_step

from svae.util import add, sub, allclose, uninterleave
from svae.lds.gaussian import pair_mean_to_natural
from svae.lds.lds_inference import lds_standard_to_natparam
from svae.lds.synthetic_data import generate_data, rand_lds
from svae.lds.lds_inference import natural_lds_inference_general, \
    natural_lds_inference_general_autograd

from lds_inference_alt import natural_filter_forward, filter_forward, \
    natural_lds_Estep, natural_lds_inference, natural_lds_inference_general_nosaving

from test_util import natural_to_mean

npr.seed(0)


### tests

def test_filters():
    def compare_filters(lds, data):
        (filtered_mus, filtered_sigmas), loglike = filter_forward(data, *lds)

        messages, lognorm = natural_filter_forward(lds_standard_to_natparam(*lds), data)
        prediction_messages, filter_messages = uninterleave(messages)
        natural_filtered_mus, natural_filtered_sigmas = zip(*map(natural_to_mean, filter_messages))

        assert all(map(np.allclose, filtered_mus, natural_filtered_mus))
        assert all(map(np.allclose, filtered_sigmas, natural_filtered_sigmas))
        assert np.isclose(loglike, lognorm)

    for _ in xrange(10):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,50)
        lds = rand_lds(n, p)
        states, data = generate_data(T, *lds)

        yield compare_filters, lds, data


def test_E_step():
    def compare_E_step(lds, data):
        natparam = lds_standard_to_natparam(*lds)
        E_init_stats, E_pairwise_stats, E_node_stats = natural_lds_Estep(natparam, data)
        E_init_stats2, E_pairwise_stats2, E_node_stats2 = pylds_E_step(lds, data)

        assert all(map(np.allclose, E_init_stats, E_init_stats2))
        assert all(map(np.allclose, E_pairwise_stats, E_pairwise_stats2))
        assert all(map(np.allclose, E_node_stats, E_node_stats2))

    for _ in xrange(10):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,50)
        lds = rand_lds(n, p)
        states, data = generate_data(T, *lds)

        yield compare_E_step, lds, data


def test_E_step_inhomog():
    def compare_E_step(lds, data):
        natparam = lds_standard_to_natparam(*lds)
        E_init_stats, E_pairwise_stats, E_node_stats = natural_lds_Estep(natparam, data)
        E_init_stats2, E_pairwise_stats2, E_node_stats2 = pylds_E_step_inhomog(lds, data)

        assert all(map(np.allclose, E_init_stats, E_init_stats2))
        assert all(map(np.allclose, E_pairwise_stats, E_pairwise_stats2))
        assert all(map(np.allclose, E_node_stats, E_node_stats2))

    for _ in xrange(10):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,50)
        lds = rand_lds(n, p, T)
        states, data = generate_data(T, *lds)

        yield compare_E_step, lds, data


def test_tuple_math_on_lds_natparam():
    def helper(homog):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,50)
        lds = rand_lds(n, p, None if homog else T)
        states, data = generate_data(T, *lds)
        natparam = lds_standard_to_natparam(*lds)
        E_stats = natural_lds_Estep(natparam, data)
        assert allclose(sub(add(natparam, E_stats), E_stats), natparam)

    yield helper, True
    yield helper, False


def test_general_inference():
    def get_general_node_params(x, lds):
        T, p = x.shape
        C, sigma_obs = lds[-2:]

        J, Jzx, Jxx, logZ = pair_mean_to_natural(C, sigma_obs)
        h = np.einsum('tzx,tx->tz', Jzx, x)
        logZ += np.einsum('ti,tij,tj->t', x, Jxx, x) - p/2.*np.log(2*np.pi)

        return J, h, logZ

    def compare_E_step(lds, data):
        natparam = init_params, pair_params, node_params = lds_standard_to_natparam(*lds)
        general_node_params = get_general_node_params(data, lds)
        C, sigma_obs = lds[-2:]
        sample, E_stats, lognorm = natural_lds_inference(natparam, data)
        sample2, E_stats2, lognorm2 = natural_lds_inference_general(
            (init_params, pair_params), general_node_params)
        sample3, E_stats3, lognorm3 = natural_lds_inference_general_nosaving(
            (init_params, pair_params), general_node_params)
        sample4, E_stats4, lognorm4 = natural_lds_inference_general_autograd(
            (init_params, pair_params), general_node_params)

        assert allclose(E_stats[:-1], E_stats2[:-1])
        assert allclose(E_stats2, E_stats3)
        assert allclose(E_stats2, E_stats4)
        assert np.isclose(lognorm, lognorm2)
        assert np.isclose(lognorm, lognorm3)
        assert np.isclose(lognorm, lognorm4)

    for _ in xrange(10):
        n, p, T = npr.randint(1, 5), npr.randint(1, 5), npr.randint(10,50)
        lds = rand_lds(n, p, T)
        states, data = generate_data(T, *lds)

        yield compare_E_step, lds, data


### util for comparing to pylds

def pylds_E_step(lds, data):
    T = data.shape[0]
    mu_init, sigma_init, A, sigma_states, C, sigma_obs = lds
    normalizer, smoothed_mus, smoothed_sigmas, E_xtp1_xtT = \
        _E_step(mu_init, sigma_init, A, sigma_states, C, sigma_obs, data)

    EyyT = data.T.dot(data)
    EyxT = data.T.dot(smoothed_mus)
    ExxT = smoothed_sigmas.sum(0) + smoothed_mus.T.dot(smoothed_mus)

    E_xt_xtT = \
        ExxT - (smoothed_sigmas[-1]
                + np.outer(smoothed_mus[-1],smoothed_mus[-1]))
    E_xtp1_xtp1T = \
        ExxT - (smoothed_sigmas[0]
                + np.outer(smoothed_mus[0], smoothed_mus[0]))
    E_xtp1_xtT = E_xtp1_xtT.sum(0)

    E_x1_x1T = smoothed_sigmas[0] + np.outer(smoothed_mus[0], smoothed_mus[0])
    E_x1 = smoothed_mus[0]

    E_init_stats = E_x1_x1T, E_x1, 1.
    E_pairwise_stats = E_xt_xtT, E_xtp1_xtT.T, E_xtp1_xtp1T, T-1
    E_node_stats = ExxT, EyxT.T, EyyT, T

    return E_init_stats, E_pairwise_stats, E_node_stats


def pylds_E_step_inhomog(lds, data):
    T = data.shape[0]
    mu_init, sigma_init, A, sigma_states, C, sigma_obs = lds
    normalizer, smoothed_mus, smoothed_sigmas, E_xtp1_xtT = \
        _E_step(mu_init, sigma_init, A, sigma_states, C, sigma_obs, data)

    EyyT = np.einsum('ti,tj->tij', data, data)
    EyxT = np.einsum('ti,tj->tij', data, smoothed_mus)
    ExxT = smoothed_sigmas + np.einsum('ti,tj->tij', smoothed_mus, smoothed_mus)

    E_xt_xtT = ExxT[:-1]
    E_xtp1_xtp1T = ExxT[1:]
    E_xtp1_xtT = E_xtp1_xtT

    E_x1_x1T = smoothed_sigmas[0] + np.outer(smoothed_mus[0], smoothed_mus[0])
    E_x1 = smoothed_mus[0]

    E_init_stats = E_x1_x1T, E_x1, 1.
    E_pairwise_stats = E_xt_xtT.sum(0), E_xtp1_xtT.sum(0).T, E_xtp1_xtp1T.sum(0), T-1
    E_node_stats = ExxT, np.transpose(EyxT, (0, 2, 1)), EyyT, np.ones(T)

    return E_init_stats, E_pairwise_stats, E_node_stats
