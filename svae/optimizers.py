import numpy as np
import numpy.random as npr
from util import add, scale, zeros_like, square, sqrt, div, add_scalar, mul, concat, \
    get_num_datapoints, split_into_batches


# TODO make optimizers into monads!
# TODO track grad statistics

def adam(data, val_and_grad, callback=None):
    num_datapoints = get_num_datapoints(data)
    def adam(allparams, nat_stepsize, stepsize, num_epochs, seq_len, num_seqs=None,
             b1=0.9, b2=0.999, eps=1e-8, num_samples=1):
        natparams, params = allparams[:1], allparams[1:]
        m = zeros_like(params)
        v = zeros_like(params)
        i = 0
        accumulate = lambda rho, a, b: add(scale(1-rho, a), scale(rho, b))

        for epoch in xrange(num_epochs):
            vals = []
            batches, num_batches = split_into_batches(data, seq_len, num_seqs)
            for y in batches:
                val, grad = scale(1./num_datapoints, val_and_grad(y, num_batches, num_samples, *allparams))
                natgrad, grad = grad[:1], grad[1:]

                m = accumulate(b1, grad, m)          # first moment estimate
                v = accumulate(b2, square(grad), v)  # second moment estimate
                mhat = scale(1./(1 - b1**(i+1)), m)  # bias correction
                vhat = scale(1./(1 - b2**(i+1)), v)
                update = scale(stepsize, div(mhat, add_scalar(eps, sqrt(vhat))))

                natparams = add(natparams, scale(nat_stepsize, natgrad))
                params = add(params, update)
                allparams = concat(natparams, params)
                vals.append(val)
                i += 1

            if callback: callback(epoch, vals, natgrad, allparams)

        return allparams
    return adam

def adadelta(data, val_and_grad, callback=None):
    num_datapoints = get_num_datapoints(data)
    def adadelta(allparams, nat_stepsize, num_epochs, seq_len, num_seqs=None,
                 rho=0.95, epsilon=1e-6, num_samples=1, permute=True):
        natparams, params = allparams[:1], allparams[1:]
        sum_gsq = zeros_like(params)  # accumulated sq. grads
        sum_usq = zeros_like(params)  # accumulated sq. updates
        accumulate = lambda a, b: add(scale(rho, a), scale(1-rho, b))

        for epoch in xrange(num_epochs):
            vals = []
            batches, num_batches = split_into_batches(data, seq_len, num_seqs)
            for y in batches:
                val, grad = scale(1./num_datapoints, val_and_grad(y, num_batches, num_samples, *allparams))
                natgrad, grad = grad[:1], grad[1:]
                sum_gsq = accumulate(sum_gsq, square(grad))
                diag_scaling = div(sqrt(add_scalar(epsilon, sum_usq)),
                                sqrt(add_scalar(epsilon, sum_gsq)))
                update = mul(diag_scaling, grad)
                sum_usq = accumulate(sum_usq, square(update))

                natparams = add(natparams, scale(nat_stepsize, natgrad))
                params = add(params, update)
                allparams = concat(natparams, params)
                vals.append(val)

            if callback: callback(epoch, vals, natgrad, allparams)
        return allparams
    return adadelta
