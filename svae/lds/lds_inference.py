from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.convenience_wrappers import grad_and_aux as agrad, value_and_grad as vgrad
from autograd.util import make_tuple
from autograd.core import primitive, primitive_with_aux
from operator import itemgetter, attrgetter

from svae.util import monad_runner, rand_psd, interleave, depth, uninterleave, \
    add, shape, zeros_like
from gaussian import mean_to_natural, pair_mean_to_natural, natural_sample, \
    natural_condition_on, natural_condition_on_general, natural_to_mean, \
    natural_rts_backward_step
# from gaussian import natural_predict, natural_lognorm
from gaussian_nochol import natural_predict, natural_lognorm

from cython_lds_inference import \
    natural_filter_forward_general as cython_natural_filter_forward, \
    natural_filter_grad as cython_natural_filter_grad, \
    natural_sample_backward as cython_natural_sample_backward, \
    natural_sample_backward_grad as cython_natural_sample_backward_grad, \
    natural_smoother_general as cython_natural_smoother_general, \
    natural_smoother_general_grad as cython_natural_smoother_grad

cython_natural_filter_forward = primitive_with_aux(cython_natural_filter_forward)
def make_natural_filter_grad_arg2(intermediates, ans, init_params, pair_params, node_params):
    return primitive(lambda g: cython_natural_filter_grad(g, intermediates))
cython_natural_filter_forward.defgrad(make_natural_filter_grad_arg2, 2)

cython_natural_sample_backward = primitive_with_aux(cython_natural_sample_backward)
def make_natural_sample_grad_arg0(intermediates, ans, messages, pair_params, num_samples):
    return primitive(lambda g: cython_natural_sample_backward_grad(g, intermediates))
cython_natural_sample_backward.defgrad(make_natural_sample_grad_arg0, 0)

cython_natural_smoother_general = primitive_with_aux(cython_natural_smoother_general)
def make_natural_smoother_grad_arg0(intermediates, ans, forward_messages, pair_params):
    return primitive(lambda g: cython_natural_smoother_grad(g, intermediates))
cython_natural_smoother_general.defgrad(make_natural_smoother_grad_arg0, 0)


### converting standard to natural parameters, plus parameter bookkeeping

def lds_standard_to_natparam(mu_init, sigma_init, A, sigma_states, C, sigma_obs):
    init_param = mean_to_natural(mu_init, sigma_init)
    pair_params = pair_mean_to_natural(A, sigma_states)
    node_params = pair_mean_to_natural(C, sigma_obs)
    return init_param, pair_params, node_params

def _repeat_param(param, length):
    # This function repeats out a parameter so that time-homogeneous models,
    # which only have one tuple of lds pair parameters, are represented as
    # time-inhomogeneous models (with a list of T lds pair parameter tuples).

    if depth(param) == 1:
        param = [param]*length
    elif len(param) != length:
        param = zip(*param)
    assert depth(param) == 2 and len(param) == length
    return param

def _canonical_init_params(init_params):
   return init_params[0], init_params[1], sum(init_params[2:])

def _canonical_node_params(node_params):
    def is_tuple_of_ndarrays(obj):
        return isinstance(obj, (tuple, list)) and \
            all(hasattr(element, 'ndim') for element in obj)

    if is_tuple_of_ndarrays(node_params):
        ndims = tuple(map(attrgetter('ndim'), node_params))
        if ndims in [(3,2,1), (3,2), (2,2,1), (2,2)]:
            T, N = node_params[1].shape
            node_params = node_params if len(node_params) == 3 else \
                node_params + (np.zeros(T),)
            allowed_shapes = [((T, N, N), (T, N), (T,)), ((T, N), (T, N), (T,))]
            if shape(node_params) in allowed_shapes:
                return node_params

    raise ValueError



### inference using info parameters and general node potentials

def natural_filter_forward_general(init_params, pair_params, node_params):
    init_params = _canonical_init_params(init_params)
    node_params = zip(*_canonical_node_params(node_params))
    pair_params = _repeat_param(pair_params, len(node_params) - 1)

    def unit(J, h, logZ):
        return [(J, h)], logZ

    def bind(result, step):
        messages, lognorm = result
        new_message, term = step(messages[-1])
        return messages + [new_message], lognorm + term

    condition = lambda node_param: lambda (J, h): natural_condition_on_general(J, h, *node_param)
    predict = lambda pair_param: lambda (J, h): natural_predict(J, h, *pair_param)
    steps = interleave(map(condition, node_params), map(predict, pair_params))

    messages, lognorm = monad_runner(bind)(unit(*init_params), steps)
    lognorm += natural_lognorm(*messages[-1])

    return messages, lognorm


def natural_sample_backward_general(forward_messages, pair_params, num_samples=None):
    filtered_messages = forward_messages[1::2]
    pair_params = _repeat_param(pair_params, len(filtered_messages) - 1)
    pair_params = map(itemgetter(0, 1), pair_params)

    unit = lambda sample: [sample]
    bind = lambda result, step: [step(result[0])] + result

    sample = lambda (J11, J12), (J_filt, h_filt): lambda next_sample: \
        natural_sample(*natural_condition_on(J_filt, h_filt, next_sample, J11, J12))
    steps = reversed(map(sample, pair_params, filtered_messages[:-1]))

    last_sample = natural_sample(*filtered_messages[-1], num_samples=num_samples)
    samples = monad_runner(bind)(unit(last_sample), steps)

    return np.concatenate([sample[None,...] for sample in samples])


def natural_smoother_general(forward_messages, init_params, pair_params, node_params):
    prediction_messages, filter_messages = uninterleave(forward_messages)
    inhomog = depth(pair_params) == 2
    T = len(prediction_messages)
    pair_params, orig_pair_params = _repeat_param(pair_params, T-1), pair_params
    node_params = zip(*_canonical_node_params(node_params))

    def unit(filtered_message):
        J, h = filtered_message
        mu, Sigma = natural_to_mean(filtered_message)
        ExxT = Sigma + np.outer(mu, mu)
        return make_tuple(J, h, mu), [(mu, ExxT, 0.)]

    def bind(result, step):
        next_smooth, stats = result
        J, h, (mu, ExxT, ExxnT) = step(next_smooth)
        return make_tuple(J, h, mu), [(mu, ExxT, ExxnT)] + stats

    rts = lambda next_pred, filtered, pair_param: lambda next_smooth: \
        natural_rts_backward_step(next_smooth, next_pred, filtered, pair_param)
    steps = reversed(map(rts, prediction_messages[1:], filter_messages[:-1], pair_params))

    _, expected_stats = monad_runner(bind)(unit(filter_messages[-1]), steps)

    def make_init_stats(a):
        mu, ExxT, _ = a
        return (ExxT, mu, 1., 1.)[:len(init_params)]

    def make_pair_stats(a, b):
        (mu, ExxT, ExnxT), (mu_n, ExnxnT, _) = a, b
        return ExxT, ExnxT.T, ExnxnT, 1.

    is_diagonal = node_params[0][0].ndim == 1
    if is_diagonal:
        def make_node_stats(a):
            mu, ExxT, _ = a
            return np.diag(ExxT), mu, 1.
    else:
        def make_node_stats(a):
            mu, ExxT, _ = a
            return ExxT, mu, 1.

    E_init_stats = make_init_stats(expected_stats[0])
    E_pair_stats = map(make_pair_stats, expected_stats[:-1], expected_stats[1:])
    E_node_stats = map(make_node_stats, expected_stats)

    if not inhomog:
        E_pair_stats = reduce(add, E_pair_stats, zeros_like(orig_pair_params))

    E_node_stats = map(np.array, zip(*E_node_stats))

    return E_init_stats, E_pair_stats, E_node_stats


### inference routines

## most general inference procedure

def natural_lds_inference_general(natparam, node_params, num_samples=None):
    init_params, pair_params = natparam
    forward_messages, lognorm = natural_filter_forward_general(
        init_params, pair_params, node_params)
    expected_stats = natural_smoother_general(
        forward_messages, init_params, pair_params, node_params)
    samples = natural_sample_backward_general(forward_messages, pair_params, num_samples)

    return samples, expected_stats, lognorm


def cython_natural_lds_inference_general(natparam, node_params, num_samples=1):
    init_params, pair_params = natparam
    forward_messages, lognorm = cython_natural_filter_forward(
        init_params, pair_params, node_params)
    expected_stats = cython_natural_smoother_general(forward_messages, pair_params)
    samples = cython_natural_sample_backward(forward_messages, pair_params, num_samples)
    return samples, expected_stats, lognorm


def natural_lds_inference_general_autograd(natparam, node_params, num_samples=None):
    init_params, pair_params = natparam

    def lds_log_normalizer(all_natparams):
        init_params, pair_params, node_params = all_natparams
        forward_messages, lognorm = natural_filter_forward_general(
            init_params, pair_params, node_params)
        return lognorm, (lognorm, forward_messages)

    all_natparams = make_tuple(init_params, pair_params, node_params)
    expected_stats, (lognorm, forward_messages) = agrad(lds_log_normalizer)(all_natparams)
    samples = natural_sample_backward_general(forward_messages, pair_params, num_samples)

    return samples, expected_stats, lognorm


## E-step for meanfield

def natural_lds_estep_general(natparam, node_params):
    init_params, pair_params = natparam
    forward_messages, lognorm = natural_filter_forward_general(
        init_params, pair_params, node_params)
    expected_stats = natural_smoother_general(
        forward_messages, init_params, pair_params, node_params)
    return lognorm, expected_stats


def cython_natural_lds_estep_general(natparam, node_params):
    init_params, pair_params = natparam
    forward_messages, lognorm = cython_natural_filter_forward(
        init_params, pair_params, node_params)
    expected_stats = cython_natural_smoother_general(forward_messages, pair_params)
    return lognorm, expected_stats


def natural_lds_estep_general_autograd(natparam, node_params):
    init_params, pair_params = natparam

    def lds_log_normalizer(natparam):
        init_params, pair_params = natparam
        _, lognorm = natural_filter_forward_general(init_params, pair_params, node_params)
        return lognorm

    return vgrad(lds_log_normalizer)(natparam)


## sampling

def natural_lds_sample(natparam, node_params, num_samples=None):
    init_params, pair_params = natparam
    forward_messages, _ = natural_filter_forward_general(
        init_params, pair_params, node_params)
    samples = natural_sample_backward_general(forward_messages, pair_params, num_samples)
    return samples

def cython_natural_lds_sample(natparam, node_params, num_samples=1):
    init_params, pair_params = natparam
    forward_messages, _ = cython_natural_filter_forward(init_params, pair_params, node_params)
    samples = cython_natural_sample_backward(forward_messages, pair_params, num_samples)
    return samples
