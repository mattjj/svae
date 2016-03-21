from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import make_tuple

from autograd import grad
from autograd.core import primitive_with_aux

from svae.lds.lds_inference import natural_filter_forward_general, \
    natural_smoother_general, natural_sample_backward_general

from svae.lds.cython_lds_inference import natural_filter_grad, \
    natural_filter_forward_general as _natural_filter_forward_general, \
    natural_smoother_general as _natural_smoother_general, \
    natural_sample_backward as _natural_sample_backward, \
    natural_sample_backward_grad, natural_smoother_general_grad

from svae.util import interleave, randn_like, contract, shape, make_unop

from test_lds_info_inference_dense import random_init_param, random_pair_param, rand_psd


_natural_filter_forward_general = primitive_with_aux(_natural_filter_forward_general)
def make_natural_filter_grad_arg2(intermediates, ans, init_params, pair_params, node_params):
    return lambda g: natural_filter_grad(g, intermediates)
_natural_filter_forward_general.defgrad(make_natural_filter_grad_arg2, 2)

_natural_sample_backward = primitive_with_aux(_natural_sample_backward)
def make_natural_sample_grad_arg0(intermediates, ans, messages, pair_params, num_samples):
    return lambda g: natural_sample_backward_grad(g, intermediates)
_natural_sample_backward.defgrad(make_natural_sample_grad_arg0, 0)

_natural_smoother_general = primitive_with_aux(_natural_smoother_general)
def make_natural_smoother_grad_arg0(intermediates, ans, forward_messages, pair_params):
    return lambda g: natural_smoother_general_grad(g, intermediates)
_natural_smoother_general.defgrad(make_natural_smoother_grad_arg0, 0)

### util

def rand_lds(n, T):
    return random_init_param(n), random_pair_param(n), random_diag_nn_potentials(n, T)

def random_diag_nn_potentials(n, T):
    return -1./2*npr.rand(T, n), npr.randn(T, n), npr.randn(T)

def allclose(m1, m2):
    if isinstance(m1, np.ndarray):
        return np.allclose(m1, m2)
    elif np.isscalar(m1):
        return np.isclose(m1, m2)
    return len(m1) == len(m2) and all(map(allclose, m1, m2))

def unpack_dense_messages(messages):
    (J_predict, h_predict), (J_filtered, h_filtered) = messages
    prediction_messages = zip(J_predict, h_predict)
    filtered_messages = zip(J_filtered, h_filtered)
    return interleave(prediction_messages, filtered_messages)

def pack_nodeparams(node_params):
    return make_tuple(*map(np.array, zip(*node_params)))

### tests

def test_filters():
    def compare_filters(lds):
        init_params, pair_params, node_params = lds
        messages1, lognorm1 = natural_filter_forward_general(
            init_params, pair_params, node_params)
        dense_messages2, lognorm2 = _natural_filter_forward_general(
            init_params, pair_params, node_params)
        messages2 = unpack_dense_messages(dense_messages2)

        assert allclose(messages1, messages2)
        assert np.isclose(lognorm1, lognorm2)

    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        yield compare_filters, rand_lds(n, T)

def test_filter_grad():
    def compare_grads(lds):
        init_params, pair_params, node_params = lds

        dotter = randn_like(natural_filter_forward_general(
            init_params, pair_params, node_params)[0])

        def messages_to_scalar(messages):
            return contract(dotter, messages)

        def py_fun(node_params):
            messages, lognorm = natural_filter_forward_general(
                init_params, pair_params, node_params)
            return np.cos(lognorm) + messages_to_scalar(messages)

        def cy_fun(node_params):
            dense_messages, lognorm = _natural_filter_forward_general(
                init_params, pair_params, node_params)
            messages = unpack_dense_messages(dense_messages)
            return np.cos(lognorm) + messages_to_scalar(messages)

        g_py = grad(py_fun)(node_params)
        g_cy = grad(cy_fun)(node_params)

        assert allclose(g_py, g_cy)

    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        yield compare_grads, rand_lds(n, T)

def test_smoothers():
    def compare_smoothers(lds):
        init_params, pair_params, node_params = lds

        messages1, _ = natural_filter_forward_general(
            init_params, pair_params, node_params)
        E_init_stats1, E_pair_stats1, E_node_stats1 = \
            natural_smoother_general(messages1, *lds)

        dense_messages2, _ = _natural_filter_forward_general(
            init_params, pair_params, node_params)
        E_init_stats2, E_pair_stats2, E_node_stats2 = \
            _natural_smoother_general(dense_messages2, pair_params)

        assert allclose(E_init_stats1[:3], E_init_stats2[:3])
        assert allclose(E_pair_stats1, E_pair_stats2)
        assert allclose(E_node_stats1, E_node_stats2)

    npr.seed(0)
    for _ in xrange(25):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        yield compare_smoothers, rand_lds(n, T)


def test_smoother_grads():
    def compare_smoother_grads(lds):
        init_params, pair_params, node_params = lds

        symmetrize = make_unop(lambda x: (x + x.T)/2. if np.ndim(x) == 2 else x, tuple)

        messages, _ = natural_filter_forward_general(*lds)
        dotter = randn_like(natural_smoother_general(messages, *lds))

        def py_fun(messages):
            result = natural_smoother_general(messages, *lds)
            assert shape(result) == shape(dotter)
            return contract(dotter, result)

        dense_messages, _ = _natural_filter_forward_general(
            init_params, pair_params, node_params)
        def cy_fun(messages):
            result = _natural_smoother_general(messages, pair_params)
            result = result[0][:3], result[1], result[2]
            assert shape(result) == shape(dotter)
            return contract(dotter, result)

        result_py = py_fun(messages)
        result_cy = cy_fun(dense_messages)
        assert np.isclose(result_py, result_cy)

        g_py = grad(py_fun)(messages)
        g_cy = unpack_dense_messages(grad(cy_fun)(dense_messages))

        assert allclose(g_py, g_cy)

    npr.seed(0)
    for _ in xrange(50):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        yield compare_smoother_grads, rand_lds(n, T)


def test_samplers():
    def compare_samplers(lds, num_samples, seed):
        init_params, pair_params, node_params = lds

        npr.seed(seed)
        messages1, _ = natural_filter_forward_general(
            init_params, pair_params, node_params)
        samples1 = natural_sample_backward_general(messages1, pair_params, num_samples)

        npr.seed(seed)
        dense_messages2, _ = _natural_filter_forward_general(
            init_params, pair_params, node_params)
        samples2 = _natural_sample_backward(dense_messages2, pair_params, num_samples)

        assert np.allclose(samples1, samples2)

    npr.seed(0)
    for i in xrange(25):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        num_samples = npr.randint(1,10)
        yield compare_samplers, rand_lds(n, T), num_samples, i

def test_sampler_grads():
    def compare_sampler_grads(lds, num_samples, seed):
        init_params, pair_params, node_params = lds

        messages, _ = natural_filter_forward_general(
            init_params, pair_params, node_params)
        def fun1(messages):
            npr.seed(seed)
            samples = natural_sample_backward_general(messages, pair_params, num_samples)
            return np.sum(np.sin(samples))
        grads1 = grad(fun1)(messages)

        messages, _ = _natural_filter_forward_general(
                init_params, pair_params, node_params)
        def fun2(messages):
            npr.seed(seed)
            samples = _natural_sample_backward(messages, pair_params, num_samples)
            return np.sum(np.sin(samples))
        grads2 = grad(fun2)(messages)

        unpack_dense_grads = lambda x: interleave(*map(lambda y: zip(*y), x))

        assert allclose(grads1, unpack_dense_grads(grads2))

    npr.seed(0)
    for i in xrange(25):
        n, T = npr.randint(1, 5), npr.randint(10, 50)
        num_samples = npr.randint(1,10)
        yield compare_sampler_grads, rand_lds(n, T), num_samples, i


if __name__ == "__main__":
    # this is here for timing
    n, T = 10, 100
    init_params, pair_params, node_params = rand_lds(n, T)

    def make_grad(fn):
        return grad(lambda node_params: fn(init_params, pair_params, node_params)[1])

    python_grad = make_grad(natural_filter_forward_general)
    cython_grad = make_grad(_natural_filter_forward_general)
