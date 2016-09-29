from __future__ import division
from autograd import value_and_grad as vgrad

def make_gradfun(run_inference, recognize, loglike, eta_prior):
    saved = lambda: None

    def mc_vlb(eta, loglike_params, recogn_params,
               data_batch, num_batches, num_samples):
        nn_potentials = recognize(recogn_params, data_batch)
        samples, saved.stats, global_vlb, local_vlb = \
            run_inference(eta_prior, eta, nn_potentials, num_samples)
        return global_vlb + num_batches * local_vlb \
            + num_batches * loglike(loglike_params, samples, data_batch)

    def gradfun(eta, loglike_params, recogn_params,
                data_batch, num_batches, num_samples):
        objective = lambda (loglike_params, recogn_params): \
            mc_vlb(eta, loglike_params, recogn_params,
                   data_batch, num_batches, num_samples)
        vlb, (loglike_grad, recogn_grad) = \
            vgrad(objective)((loglike_params, recogn_params))
        eta_natgrad = eta_prior + num_batches * saved.stats - eta
        return vlb, (eta_natgrad, loglike_grad, recogn_grad)

    return gradfun
