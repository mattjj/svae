from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import operator as op

from svae.svae import make_gradfun
from svae.optimizers import adam
from svae.recognition_models import linear_recognize, init_linear_recognize
from svae.forward_models import linear_loglike, init_linear_loglike
from svae.util import add, scale, randn_like, flatten, rle, make_unop

from svae.models.slds_svae import run_inference, make_slds_global_natparam
import svae.lds.mniw as mniw

random_rotation = lambda N: np.linalg.qr(npr.randn(N, N))[0]

rot = lambda theta: \
    np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

def evenly_spaced_rotations(N, K, scale):
    if N != 2: raise ValueError
    omegas = np.linspace(0, np.pi / 2., K, endpoint=False)
    return [scale*rot(omega) for omega in omegas]

def generate_synthetic_data(K, N, P):
    z = np.tile(np.arange(K).repeat(100), 10)
    T = len(z)

    As = evenly_spaced_rotations(N, K, 0.95)
    # As = [0.9 * random_rotation(N) for k in range(K)]

    x = np.zeros((T, N))
    x[0] = np.eye(N)[0]
    eps = 0.1*npr.randn(T, N)
    for t in range(T-1):
        x[t+1] = np.dot(As[z[t+1]], x[t]) + eps[t]

    C = np.eye(N)
    y = np.dot(x, C.T) + 0.1 * npr.randn(T, P)

    return y, (As, 0.1**2 * np.eye(N), C, 0.1**2 * np.eye(P))  # noises are inflated

def make_slds_global_natparam_from_truth(As, Bs):
    def lds_pair_natparam(A, B):
        N = A.shape[0]
        nu = 5.
        S = nu * B
        M = A
        K = 1./nu * np.eye(N)
        return mniw.standard_to_natural(nu, S, M, K)

    K, N = len(As), len(As[0])
    hmm_natparam, _ = make_slds_global_natparam(K, N, alpha=3., sticky_bias=10., random=True)
    init_natparam = 2*np.eye(N), np.zeros(N), 2., 2.

    return hmm_natparam, [(init_natparam, lds_pair_natparam(A, B)) for A, B in zip(As, Bs)]

def param_bro(As, Bs):
    def replace_lds_natparam(lds_natparam):
        init_natparam, pair_natparam = lds_natparam
        nu, S, M, K = mniw.natural_to_standard(pair_natparam)
        # new_M = M
        # new_M = npr.randn(*M.shape)
        new_M = M + 0.2*npr.randn(*M.shape)
        return init_natparam, mniw.standard_to_natural(nu, S, new_M, K)

    natparam = make_slds_global_natparam_from_truth(As, Bs)
    hmm_natparam, lds_natparams = natparam
    return hmm_natparam, map(replace_lds_natparam, lds_natparams)

def _zero(arr):
    arr = np.copy(arr)
    arr[120:140] = 0
    return arr
zero = make_unop(_zero, tuple)

def show_states(params, data):
    from svae.models.slds_svae import optimize_local_meanfield

    natparam, phi, psi = params
    hmm_global_natparam, lds_global_natparam = natparam

    node_potentials = zero(linear_recognize(data, psi))
    (hmm_stats, _), _, _ = optimize_local_meanfield(natparam, node_potentials)

    pcolor_states(data, hmm_stats[2].argmax(1))

def pcolor_states(data, stateseq, cmap=plt.cm.Set1):
    fig, ax = plt.subplots()

    stateseq_norep, durations = rle(stateseq)
    datamin, datamax = data.min(), data.max()

    x, y = np.hstack((0, durations.cumsum())), np.array((datamin, datamax))
    # C = np.atleast_2d([cmap(state) for state in stateseq_norep])[:,None,:]
    C = np.atleast_2d(stateseq_norep)

    ax.plot(data)
    ax.pcolorfast(x, y, C, vmin=0, vmax=1, alpha=0.3)
    ax.set_ylim((datamin, datamax))
    ax.set_xlim((0, data.shape[0]))

if __name__ == "__main__":
    npr.seed(1)

    # model size parameters
    K = 2  # number of states
    N = 2  # latent state dimension
    P = 2  # observation dimension

    # generate synthetic data
    data, (As, B, C, D) = generate_synthetic_data(K, N, P)
    T = data.shape[0]
    plt.plot(data)
    # plt.savefig('synth_data.png')
    # plt.close()

    # set prior natparam
    # prior_natparam2 = make_slds_global_natparam(K, N, alpha=2., sticky_bias=0., random=True)
    # prior_natparam = make_slds_global_natparam_from_truth(As, [B]*K)
    prior_natparam = param_bro(As, [B]*K)

    # build svae gradient function
    gradfun = make_gradfun(run_inference, linear_recognize, linear_loglike, prior_natparam)

    # set up optimizer
    def callback(itr, vals, natgrad, params):
        print('{}: {} (natgrad_mean={}, natgrad_std={})'.format(
            itr, np.mean(vals), np.mean(flatten(natgrad)), np.std(flatten(natgrad))))
        show_states(params, data[:400])
        plt.savefig('slds_svae_synth_{}.png'.format(itr))
        plt.close()
    optimize = adam(data, gradfun, callback)

    # set prior to something generic, initilize estimated params
    init_phi, init_psi = init_linear_loglike(N, P), init_linear_recognize(N, P)
    params = prior_natparam, init_phi, init_psi

    # optimize
    params = optimize(params, 1e-1, 1e-2, num_epochs=100, seq_len=200)

    # plot results
    show_states(params, data[:400])

    (hmm_natparam, lds_natparams), _, _ = params
    lds_pair_params = map(op.itemgetter(1), lds_natparams)
    fit_As, fit_Sigmas = zip(*map(mniw.expected_standard_params, lds_pair_params))

    for A in fit_As: print(np.linalg.eigvals(A))
    print('')
    for A in As: print(np.linalg.eigvals(A))

    plt.show()
