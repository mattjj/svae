from __future__ import division
from toolz import curry
from autograd import value_and_grad as vgrad
from autograd.util import flatten
from svae.util import flat, split_into_batches

@curry
def make_gradfun(run_inference, recognize, loglike, pgm_prior, data,
                 batch_size, num_samples, natgrad_scale=1.):
    pgm_prior, unflat = flatten(pgm_prior)
    data_batches, num_batches = split_into_batches(data, batch_size)

    saved = lambda: None

    def mc_elbo(pgm_params, loglike_params, recogn_params, i):
        nn_potentials = recognize(recogn_params, data_batches[i])
        samples, saved.stats, global_kl, local_kl = \
            run_inference(pgm_prior, pgm_params, nn_potentials, num_samples)
        return num_batches * loglike(loglike_params, samples, data_batches[i]) \
            - global_kl - num_batches * local_kl

    def gradfun(params, i):
        pgm_params, loglike_params, recogn_params = params
        objective = lambda (loglike_params, recogn_params): \
            -mc_elbo(pgm_params, loglike_params, recogn_params, i)
        val, (loglike_grad, recogn_grad) = vgrad(objective)((loglike_params, recogn_params))
        pgm_natgrad = unflat(natgrad_scale * (pgm_prior + num_batches*flat(stats) - flat(pgm_params)))
        return val, (pgm_natgrad, loglike_grad, recogn_grad)

    return gradfun
