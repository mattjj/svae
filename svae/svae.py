from __future__ import division
from autograd import value_and_grad as vgrad
from autograd.util import flatten
from toolz import curry

flat = lambda x: flatten(x)[0]

@curry
def make_gradfun(run_inference, recognize, loglike, pgm_prior, data,
                 batch_size, num_samples, natgrad_scale=1.):
    pgm_prior_flat, unflatten = flatten(pgm_prior)
    # TODO split up data here

    saved = lambda: None

    def mc_elbo(pgm_params, loglike_params, recogn_params, i):
        nn_potentials = recognize(recogn_params, data_batch(i))
        samples, saved.stats, global_kl, local_kl = \
            run_inference(pgm_prior, pgm_params, nn_potentials, num_samples)
        return global_kl + num_batches * local_kl \
            + num_batches * loglike(loglike_params, samples, data_batch)

    def gradfun(params, i):
        pgm_params, loglike_params, recogn_params = params
        objective = lambda (loglike_params, recogn_params): \
            -mc_elbo(pgm_params, loglike_params, recogn_params, i)
        vlb, (loglike_grad, recogn_grad) = vgrad(objective)((loglike_params, recogn_params))
        pgm_natgrad = unflatten(pgm_prior_flat + num_batches*flat(stats) - flat(pgm_params))
        return vlb, (pgm_natgrad, loglike_grad, recogn_grad)

    return gradfun
