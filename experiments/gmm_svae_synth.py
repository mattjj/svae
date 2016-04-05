from __future__ import division, print_function
import numpy as np
import numpy.random as npr

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import linear_recognize, init_linear_recognize
from svae.forward_models import linear_loglike, init_linear_loglike

from svae.models.gmm_svae import run_inference, make_gmm_global_natparam


if __name__ == "__main__":
    npr.seed(0)

    K = 2  # number of components in mixture model
    N = 2  # number of latent dimensions
    P = 2  # number of observation dimensions

    # generate synthetic data
    data = np.concatenate([npr.randn(50, P), 2 + npr.randn(100, P)])

    # set prior natparam
    prior_natparam = make_gmm_global_natparam(K, N, alpha=2., random=True)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, linear_recognize, linear_loglike, prior_natparam)

    # set up optimizer
    def callback(itr, vals, natgrad, params):
        print('{}: {}'.format(itr, np.mean(vals)))
    optimize = adam(data, gradfun, callback)

    # set initialization to something generic
    init_phi, init_psi = init_linear_loglike(N, P), init_linear_recognize(N, P)
    params = prior_natparam, init_phi, init_psi

    # optimize
    params = optimize(params, 1e-2, 1e-3, num_epochs=100, seq_len=50)
