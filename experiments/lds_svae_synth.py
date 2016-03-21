from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import linear_recognize
from svae.forward_models import linear_loglike
from svae.util import add, scale, randn_like, make_unop
import svae.lds.mniw as mniw

# from svae.models.lds_svae import run_inference
from svae.models.lds_svae import cython_run_inference as run_inference
from svae.models.lds_svae import make_prior_natparam, \
    generate_test_model_and_data, linear_recognition_params_from_lds

from svae.models.lds_svae import lds_prior_expectedstats
from svae.lds.lds_inference import cython_natural_lds_sample as natural_lds_sample

rot = lambda theta: \
    np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

def generate_data(A, B, C, D, T):
    P, N = C.shape
    x = np.zeros((T, N))
    eps = npr.randn(T-1, N)

    x[0] = npr.randn(N)
    for t in range(T-1):
        x[t+1] = np.dot(A, x[t]) + np.dot(B, eps[t])

    y = np.dot(x, C.T) + np.dot(npr.randn(T, P), D.T)

    return y

def matshow(A):
    print(A)
    # plt.matshow(A, cmap='plasma')
    # plt.axis('off')
    # plt.savefig('lds_synth_fit.png')
    # plt.close()

def _zero(x):
    sl = slice(len(x)//3, 2*(len(x)//3))
    out = np.copy(x)
    out[sl] = 0
    return out
zero = make_unop(lambda x: _zero(x), tuple)

def sample_states(params, data, num_samples=1):
    natparam, phi, psi = params
    local_natparam = lds_prior_expectedstats(natparam)
    node_potentials = zero(linear_recognize(data, psi))
    return natural_lds_sample(local_natparam, node_potentials, num_samples=1).mean(1)

def sample(params, data, num_samples=1):
    _, _, psi = params
    C, D = phi
    return np.dot(sample_states(params, data, num_samples), C.T)

if __name__ == "__main__":
    npr.seed(0)

    # size parameters
    N = 2
    P = 2
    T = 900

    # generate synthetic data
    A = 0.999 * rot(2*np.pi / 30)
    # A = np.array([[0.9, 1], [0., 0.9]])
    B = 0.1 * np.eye(N)
    C = np.eye(N)
    D = 0.0001 * np.eye(N)
    phi = psi = (C, D)

    data = generate_data(A, B, C, D, T)

    plt.figure()
    plt.plot(data)

    # set up prior
    prior_natparam = make_prior_natparam(N)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, linear_recognize, linear_loglike, prior_natparam)

    # set up optimizer
    def callback(itr, vals, natgrad, params):
        print('{}: {}'.format(itr, np.mean(vals)))
    optimize = adam(data, gradfun, callback)

    # initialize estimated params to truth plus a bit of noise
    # init_phi = add(phi, scale(0.1, randn_like(phi)))
    # init_psi = add(psi, scale(0.1, randn_like(psi)))
    init_phi, init_psi = phi, psi
    params = prior_natparam, init_phi, init_psi

    # optimize
    params = optimize(params, 1e-2, 0., num_epochs=300, seq_len=T)

    natparam, _, _ = params
    niw_natparam, mniw_natparam = natparam
    expected_A, expected_Sigma = mniw.expected_standard_params(mniw_natparam)
    sampled_A, sampled_Sigma = mniw.natural_sample(mniw_natparam)
    matshow(np.vstack((A, expected_A, sampled_A)))
    matshow(np.vstack((D**2, expected_Sigma, sampled_Sigma)))

    plt.figure()
    plt.plot(sample(params, data))

    plt.show()
