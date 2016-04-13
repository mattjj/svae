from __future__ import division
from autograd import value_and_grad as vgrad
from util import add, sub, contract, scale, unbox


def make_gradfun(run_inference, recognize, loglike, eta_prior, return_flat=False):
    saved = lambda: None

    def mc_vlb(eta, phi, psi, y_n, N, L):
        T = y_n.shape[0]
        nn_potentials = recognize(y_n, psi)
        samples, stats, global_vlb, local_vlb = run_inference(
            eta_prior, eta, nn_potentials, num_samples=L)
        saved.stats = scale(N, unbox(stats))
        return global_vlb + N * (local_vlb + loglike(y_n, samples, phi))

    def gradfun(y_n, N, L, eta, phi, psi):
        objective = lambda (phi, psi): mc_vlb(eta, phi, psi, y_n, N, L)
        vlb, (phi_grad, psi_grad) = vgrad(objective)((phi, psi))
        eta_natgrad = sub(add(eta_prior, saved.stats), eta)
        return vlb, (eta_natgrad, phi_grad, psi_grad)

    def flat_gradfun(y_n, N, L, eta, phi, psi):
        objective = lambda (eta, phi, psi): mc_vlb(eta, phi, psi, y_n, N, L)
        return vgrad(objective)((eta, phi, psi))

    if return_flat:
        return gradfun, flat_gradfun

    return gradfun
