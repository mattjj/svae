from __future__ import division
import autograd.numpy as np
from autograd import grad
from operator import itemgetter

from svae.util import monad_runner, interleave, uninterleave
from svae.lds.gaussian import sample, predict, condition_on
from svae.lds.gaussian import natural_sample, \
    natural_condition_on, natural_rts_backward_step
from svae.lds.gaussian_nochol import natural_predict, natural_lognorm
from svae.lds.lds_inference import _repeat_param, natural_filter_forward_general, \
    natural_sample_backward_general


def _unpack_repeated(natparam, T=None):
    'handle homogeneous models by repeating natural parameters if necessary'
    init_params, pair_params, node_params = natparam
    T = len(node_params) if T is None else T
    return init_params, _repeat_param(pair_params, T-1), _repeat_param(node_params, T)


### inference using standard parameters

def filter_forward(data, mu_init, sigma_init, A, sigma_states, C, sigma_obs):
    def observe(y):
        def update_belief(mu, sigma):
            mu_pred, sigma_pred = predict(mu, sigma, A, sigma_states)
            (mu_filt, sigma_filt), ll = condition_on(mu_pred, sigma_pred, C, y, sigma_obs)
            return (mu_filt, sigma_filt), ll
        return update_belief

    def unit(mu, sigma):
        return ([mu], [sigma]), 0.

    def bind(result, step):
        (mus, sigmas), lognorm = result
        (mu, sigma), term = step(mus[-1], sigmas[-1])
        return (mus + [mu], sigmas + [sigma]), lognorm + term

    (mu_filt, sigma_filt), ll = condition_on(mu_init, sigma_init, C, data[0], sigma_obs)
    (filtered_mus, filtered_sigmas), loglike = \
        monad_runner(bind)(unit(mu_filt, sigma_filt), map(observe, data[1:]))

    return (filtered_mus, filtered_sigmas), loglike + ll


def sample_backward(filtered_mus, filtered_sigmas, A, sigma_states):
    def filtered_sampler(mu_filt, sigma_filt):
        def sample_cond(next_state):
            (mu_cond, sigma_cond), _ = condition_on(
                mu_filt, sigma_filt, A, next_state, sigma_states)
            return sample(mu_cond, sigma_cond)
        return sample_cond

    def unit(sample):
        return [sample]

    def bind(result, step):
        samples = result
        sample = step(samples[0])
        return [sample] + samples

    last_sample = sample(filtered_mus[-1], filtered_sigmas[-1])
    steps = reversed(map(filtered_sampler, filtered_mus[:-1], filtered_sigmas[:-1]))
    samples = monad_runner(bind)(unit(last_sample), steps)

    return np.array(samples)


def sample_lds(data, mu_init, sigma_init, A, sigma_states, C, sigma_obs):
    (filtered_mus, filtered_sigmas), loglike = filter_forward(
        data, mu_init, sigma_init, A, sigma_states, C, sigma_obs)
    sampled_states = sample_backward(
        filtered_mus, filtered_sigmas, A, sigma_states)
    return sampled_states, loglike

### inference using info parameters and linear node potentials

def natural_filter_forward(natparam, data):
    T, p = data.shape
    init_params, pair_params, node_params = _unpack_repeated(natparam, T)

    def unit(J, h):
        return [(J, h)], 0.

    def bind(result, step):
        messages, lognorm = result
        new_message, term = step(messages[-1])
        return messages + [new_message], lognorm + term

    condition = lambda node_param, y: lambda (J, h): natural_condition_on(J, h, y, *node_param)
    predict = lambda pair_param: lambda (J, h): natural_predict(J, h, *pair_param)
    steps = interleave(map(condition, node_params, data), map(predict, pair_params))

    J_init, h_init, logZ_init = init_params
    messages, lognorm = monad_runner(bind)(unit(J_init, h_init), steps)
    lognorm += natural_lognorm(*messages[-1]) + logZ_init

    return messages, lognorm - T*p/2*np.log(2*np.pi)


def natural_smooth_backward(forward_messages, natparam):
    prediction_messages, filter_messages = uninterleave(forward_messages)
    init_params, pair_params, node_params = _unpack_repeated(natparam)
    pair_params = map(itemgetter(0, 1, 2), pair_params)

    unit = lambda (J, h): [(J, h)]
    bind = lambda result, step: [step(result[0])] + result

    rts = lambda next_prediction, filtered, pair_param: lambda next_smoothed: \
        natural_rts_backward_step(next_smoothed, next_prediction, filtered, pair_param)
    steps = map(rts, prediction_messages[1:], filter_messages, pair_params)

    return map(itemgetter(2), monad_runner(bind)(unit(filter_messages[-1]), steps))


def natural_lds_Estep(natparam, data):
    log_normalizer = lambda natparam: natural_filter_forward(natparam, data)[1]
    return grad(log_normalizer)(natparam)


def natural_lds_inference(natparam, data):
    saved = lambda: None

    def lds_log_normalizer(natparam):
        saved.forward_messages, saved.lognorm = natural_filter_forward(natparam, data)
        return saved.lognorm

    expected_stats = grad(lds_log_normalizer)(natparam)
    sample = natural_sample_backward(saved.forward_messages, natparam)

    return sample, expected_stats, saved.lognorm


def natural_sample_backward(forward_messages, natparam):
    _, filter_messages = uninterleave(forward_messages)
    _, pair_params, _ = _unpack_repeated(natparam, len(filter_messages))
    pair_params = map(itemgetter(0, 1), pair_params)

    unit = lambda sample: [sample]
    bind = lambda result, step: [step(result[0])] + result

    sample = lambda (J11, J12), (J_filt, h_filt): lambda next_sample: \
        natural_sample(*natural_condition_on(J_filt, h_filt, next_sample, J11, J12))
    steps = reversed(map(sample, pair_params, filter_messages[:-1]))

    last_sample = natural_sample(*filter_messages[-1])
    samples = monad_runner(bind)(unit(last_sample), steps)

    return np.array(samples)


def natural_lds_sample(natparam, data):
    forward_messages, lognorm = natural_filter_forward(natparam, data)
    sample = natural_sample_backward(forward_messages, natparam)
    return sample


### slightly less efficient method for testing against main method

def natural_lds_inference_general_nosaving(natparam, node_params):
    init_params, pair_params = natparam

    def lds_log_normalizer(all_natparams):
        init_params, pair_params, node_params = all_natparams
        forward_messages, lognorm = natural_filter_forward_general(init_params, pair_params, node_params)
        return lognorm

    all_natparams = init_params, pair_params, node_params
    expected_stats = grad(lds_log_normalizer)(all_natparams)
    forward_messages, lognorm = natural_filter_forward_general(init_params, pair_params, node_params)
    sample = natural_sample_backward_general(forward_messages, pair_params)

    return sample, expected_stats, lognorm
