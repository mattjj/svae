from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr

from svae.util import sub, contract, unbox

from svae.lds.lds_inference import natural_lds_inference_general, \
    cython_natural_lds_inference_general
from svae.lds.synthetic_data import generate_data, rand_lds
from svae.lds.gaussian import pair_mean_to_natural
from svae.lds import niw, mniw


### the LDS prior is a product: NIW on initial state distn, MNIW on transition distn

def lds_prior_kl(global_natparam, prior_natparam, expected_stats=None):
    if expected_stats is None:
        expected_stats = lds_prior_expectedstats(global_natparam)
    return contract(sub(prior_natparam, global_natparam), expected_stats) \
        - (lds_prior_logZ(prior_natparam) - lds_prior_logZ(global_natparam))


def lds_prior_expectedstats(natparam):
    niw_natparam, mniw_natparam = natparam
    return niw.expectedstats(niw_natparam), mniw.expectedstats(mniw_natparam)


def lds_prior_logZ(natparam):
    niw_natparam, mniw_natparam = natparam
    return niw.logZ(niw_natparam) + mniw.logZ(mniw_natparam)


### build inference function

def python_run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    local_natparam = lds_prior_expectedstats(global_natparam)
    samples, expected_stats, local_normalizer = natural_lds_inference_general(
        local_natparam, nn_potentials, num_samples)
    global_expected_stats, local_expected_stats = expected_stats[:-1], expected_stats[-1]
    local_kl = contract(nn_potentials, local_expected_stats[:2]) - local_normalizer
    global_kl = lds_prior_kl(global_natparam, prior_natparam, local_natparam)
    return samples, global_expected_stats, global_kl, local_kl


def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    local_natparam = lds_prior_expectedstats(global_natparam)
    samples, expected_stats, local_normalizer = cython_natural_lds_inference_general(
        local_natparam, nn_potentials, num_samples)
    global_expected_stats, local_expected_stats = expected_stats[:-1], expected_stats[-1]
    local_kl = contract(nn_potentials, local_expected_stats[:2]) - local_normalizer
    global_kl = lds_prior_kl(global_natparam, prior_natparam, local_natparam)
    return samples, global_expected_stats, global_kl, local_kl


def make_encoder_decoder(recognize, decode):
    def encode_mean(data, pgm_params, recogn_params):
        nn_potentials = recognize(recogn_params, data)
        _, expected_stats, _ = cython_natural_lds_inference_general(
            local_natparam, nn_potentials, 0)
        return expected_stats[-1][-1]

    def decode_mean(x, loglike_params):
        mu, _ = decode(loglike_params, x)
        return mu

    return encode_mean, decode_mean


### convenient for testing

def init_pgm_param(n, random=False, scaling=1.):
    if random: raise NotImplementedError

    nu, S, mu, kappa = n+1., 2.*scaling*(n+1)*np.eye(n), np.zeros(n), 1./(2.*scaling*n)
    # M, K = np.zeros((n,n)), 1./(2.*scaling*n)*np.eye(n)
    M, K = np.eye(n), 1./(2.*scaling*n)*np.eye(n)

    init_state_prior_natparam = niw.standard_to_natural(nu, S, mu, kappa)
    dynamics_prior_natparam = mniw.standard_to_natural(nu, S, M, K)

    return init_state_prior_natparam, dynamics_prior_natparam


def generate_test_model_and_data(n, p, T):
    lds = rand_lds(n, p)
    _, data = generate_data(T, *lds)
    return lds, data


def linear_recognition_params_from_lds(lds):
    C, sigma_obs = lds[-2:]
    return C, np.linalg.cholesky(sigma_obs)
