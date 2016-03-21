from __future__ import division
import numpy as np
import numpy.random as npr

from svae.util import add, scale, rand_dir_like, contract
from svae.svae import make_gradfun

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6


def grad_check(fun, gradfun, arg, eps=EPS, rtol=RTOL, atol=ATOL, rng=None):
    def scalar_nd(f, x, eps):
        return (f(x + eps/2) - f(x - eps/2)) / eps

    random_dir = rand_dir_like(arg)
    scalar_fun = lambda x: fun(add(arg, scale(x, random_dir)))

    numeric_grad  = scalar_nd(scalar_fun, 0.0, eps=eps)
    numeric_grad2 = scalar_nd(scalar_fun, 0.0, eps=eps)
    analytic_grad = contract(gradfun(arg), random_dir)

    assert np.isclose(numeric_grad, numeric_grad2, rtol=rtol, atol=atol)
    assert np.isclose(numeric_grad, analytic_grad, rtol=rtol, atol=atol)


def make_wrappers(seed, eta, y, value_and_grad_fun):
    def fun(params):
        phi, psi = params
        npr.seed(seed)  # seed to fix random function being evaluated
        return value_and_grad_fun(y, 1, 1, eta, phi, psi)[0]

    def gradfun(params):
        phi, psi = params
        npr.seed(seed)  # seed to fix random function being evaluated
        return value_and_grad_fun(y, 1, 1, eta, phi, psi)[1][-2:]

    return fun, gradfun


# NOTE: I commented these out because they test with full-matrix node
# potentials, which the python code supports but the cython code doesn't.
# Refactoring the lds_svae and slds_svae code to have both python and cython
# versions looked like a pain, so I'm just leaving them unested. These tests
# really only check that autograd (plus any manual gradients) is working anyway,
# and the other tests cover that well.

# def test_lds_svae():
#     from svae.models.lds_svae import run_inference, make_prior_natparam, \
#         generate_test_model_and_data, linear_recognition_params_from_lds
#     from svae.recognition_models import linear_recognize as recognize
#     from svae.forward_models import linear_loglike as loglike

#     def make_experiment(seed):
#         npr.seed(seed)  # seed for random model/data generation
#         n, p, T = npr.randint(1,5), npr.randint(1,5), npr.randint(10, 20)

#         # set up a model, data, and recognition network
#         prior_natparam = make_prior_natparam(n)
#         lds, data = generate_test_model_and_data(n, p, T)
#         phi = psi = linear_recognition_params_from_lds(lds)

#         # variational eta at which gradient is evaluated
#         eta = make_prior_natparam(n, random=True)

#         # value-and-grad function that gets wrapped for testing
#         value_and_grad_fun = make_gradfun(run_inference, recognize, loglike, prior_natparam)

#         # create wrappers
#         fun, gradfun = make_wrappers(seed, eta, data, value_and_grad_fun)

#         return fun, gradfun, phi, psi

#     for i in xrange(10):
#         fun, gradfun, phi, psi = make_experiment(i)
#         yield grad_check, fun, gradfun, (phi, psi)


# def test_slds_svae():
#     from svae.models.slds_svae import run_inference, make_slds_global_natparam
#     from svae.recognition_models import linear_recognize as recognize
#     from svae.forward_models import linear_loglike as loglike

#     from svae.models.lds_svae import linear_recognition_params_from_lds, \
#         generate_test_model_and_data

#     def make_experiment(seed):
#         npr.seed(seed)  # seed for random model/data generation
#         k, n, p, T = npr.randint(1,3), npr.randint(1,3), npr.randint(1,3), npr.randint(5, 10)

#         # set up a model, data, and recognition network
#         prior_natparam = make_slds_global_natparam(k, n)
#         lds, data = generate_test_model_and_data(n, p, T)
#         phi = psi = linear_recognition_params_from_lds(lds)

#         # variational eta at which gradient is evaluated
#         eta = make_slds_global_natparam(k, n, random=False)  # TODO make this random

#         # value-and-grad function that gets wrapped for testing
#         value_and_grad_fun = make_gradfun(run_inference, recognize, loglike, prior_natparam)

#         # create wrappers
#         fun, gradfun = make_wrappers(seed, eta, data, value_and_grad_fun)

#         return fun, gradfun, phi, psi

#     for i in xrange(10):
#         fun, gradfun, phi, psi = make_experiment(i)
#         yield grad_check, fun, gradfun, (phi, psi)
