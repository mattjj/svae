from __future__ import division, print_function
from toolz import curry
from autograd import value_and_grad as vgrad
from autograd.util import flatten
from util import split_into_batches, get_num_datapoints

callback = lambda i, val, params, grad: print('{}: {}'.format(i, val))

@curry
def make_gradfun(run_inference, recognize, loglike, pgm_prior, data,
                 batch_size, num_samples, natgrad_scale=1., callback=callback):
    _, unflat = flatten(pgm_prior)
    flat = lambda struct: flatten(struct)[0]
    num_datapoints = get_num_datapoints(data)
    data_batches, num_batches = split_into_batches(data, batch_size)
    get_batch = lambda i: data_batches[i % num_batches]
    saved = lambda: None

    def mc_elbo(pgm_params, loglike_params, recogn_params, i):
        nn_potentials = recognize(recogn_params, get_batch(i))
        samples, saved.stats, global_kl, local_kl = \
            run_inference(pgm_prior, pgm_params, nn_potentials, num_samples)
        return (num_batches * loglike(loglike_params, samples, get_batch(i))
                - global_kl - num_batches * local_kl) / num_datapoints

    def gradfun(params, i):
        pgm_params, loglike_params, recogn_params = params
        objective = lambda (loglike_params, recogn_params): \
            -mc_elbo(pgm_params, loglike_params, recogn_params, i)
        val, (loglike_grad, recogn_grad) = vgrad(objective)((loglike_params, recogn_params))
        pgm_natgrad = -natgrad_scale / num_datapoints * \
            (flat(pgm_prior) + num_batches*flat(saved.stats) - flat(pgm_params))
        grad = unflat(pgm_natgrad), loglike_grad, recogn_grad
        if callback: callback(i, val, params, grad)
        return grad

    return gradfun
