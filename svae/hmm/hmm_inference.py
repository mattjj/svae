from __future__ import division
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad as vgrad
from autograd.scipy.misc import logsumexp
from autograd.core import primitive_with_aux, primitive

from pyhsmm.internals.hmm_messages_interface import \
    messages_backwards_log, messages_forwards_log, expected_statistics_log, \
    viterbi as _viterbi

from cython_hmm_inference import hmm_logZ, hmm_logZ_grad

hmm_logZ = primitive_with_aux(hmm_logZ)
def make_grad_hmm_logZ(intermediates, ans, hmm):
    _, pair_params, _ = hmm
    return primitive(lambda g: hmm_logZ_grad(g, intermediates))
hmm_logZ.defgrad(make_grad_hmm_logZ)


def hmm_estep(natparam):
    C = lambda x: np.require(x, np.double, 'C')
    init_params, pair_params, node_params = map(C, natparam)

    # compute messages
    alphal = messages_forwards_log(
        np.exp(pair_params), node_params, np.exp(init_params),
        np.zeros_like(node_params))
    betal = messages_backwards_log(
        np.exp(pair_params), node_params,
        np.zeros_like(node_params))

    # compute expected statistics from messages
    expected_states, expected_transcounts, log_normalizer = \
        expected_statistics_log(
            pair_params, node_params, alphal, betal,
            np.zeros_like(node_params), np.zeros_like(pair_params))

    expected_stats = expected_states[0], expected_transcounts, expected_states

    return log_normalizer, expected_stats


def hmm_logZ_python(natparam):
    init_params, pair_params, node_params = natparam

    log_alpha = init_params + node_params[0]
    for node_param in node_params[1:]:
        log_alpha = logsumexp(log_alpha[:,None] + pair_params, axis=0) + node_param

    return logsumexp(log_alpha)


def hmm_viterbi(natparam):
    init_params, pair_params, node_params = natparam
    T = node_params.shape[0]

    C = lambda x: np.require(x, requirements='C')
    pair_params, node_params, init_params = \
        C(np.exp(pair_params)), C(node_params), C(np.exp(init_params))

    return _viterbi(pair_params, node_params, init_params,
                    np.zeros(T, dtype=np.int32))

hmm_estep_slow = vgrad(hmm_logZ)
