from __future__ import division, print_function
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam, sgd
from svae.svae import make_gradfun
from svae.nnet import init_mlp, make_loglike, gaussian_mean, gaussian_info
from svae.models.lds import run_inference, init_pgm_param, make_encoder_decoder

# these imports are for plotting
from svae.models.lds import lds_prior_expectedstats
from svae.lds.lds_inference import cython_natural_lds_sample as natural_lds_sample
from svae.util import zeros_like, make_unop

### data generation

triangle = lambda t: sawtooth(np.pi*t, width=0.5)
make_dot_trajectory = lambda x0, v: lambda t: triangle(v*(t + (1+x0)/2.))
make_renderer = lambda grid, sigma: lambda x: np.exp(-1./2 * (x - grid)**2/sigma**2)

def make_dot_data(image_width, T, num_steps, x0=0.0, v=0.5, render_sigma=0.2, noise_sigma=0.1):
    grid = np.linspace(-1, 1, image_width, endpoint=True)
    render = make_renderer(grid, render_sigma)
    x = make_dot_trajectory(x0, v)
    images = np.vstack([render(x(t)) for t in np.linspace(0, T, num_steps)])
    return images + noise_sigma * npr.randn(*images.shape)

### plotting

zero_after_prefix = lambda prefix: make_unop(lambda x: np.concatenate(
    (x[:prefix], np.zeros_like(x[prefix:]))), tuple)

def make_plotter(recognize, decode, data, params, prefix, plot_every):
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()

    data = data[:200]

    def generate_samples(params, data, prefix):
        _, loglike_params, _ = params
        x = sample_conditional_states(params, data, zero_after=prefix)
        return decode_mean(x, loglike_params)

    def sample_conditional_states(params, data, zero_after=-1):
        pgm_params, loglike_params, recogn_params = params
        zero = zero_after_prefix(zero_after)
        node_potentials = zero(recognize(recogn_params, data))
        local_natparam = lds_prior_expectedstats(pgm_params)
        x = natural_lds_sample(local_natparam, node_potentials, num_samples=50)
        return x

    def plot(i, val, params, grad):
        print('{}: {}'.format(i, val))
        y = generate_samples(params, data, prefix)
        T, num_samples, ndim = y.shape

        mean_image = y.mean(1)
        sample_images = np.hstack([y[:,i,:] for i in npr.choice(num_samples, 5, replace=False)])
        big_image = np.hstack((data, mean_image, sample_images))

        ax.matshow(big_image, cmap='gray')
        ax.autoscale(False)
        ax.axis('off')
        ax.plot([-0.5, big_image.shape[1]], [prefix-0.5, prefix-0.5], 'r', linewidth=2)

        fig.tight_layout()
        fig.savefig('dots_{:03d}.png'.format(i))
        plt.close('all')

    return plot

if __name__ == '__main__':
    npr.seed(0)
    plt.ion()

    # latent space dimension
    N = 10

    # generate data
    data = make_dot_data(20, 500, 5000, v=0.75, render_sigma=0.15, noise_sigma=0.1)
    T, P = data.shape

    # set up prior
    pgm_prior_params = init_pgm_param(N)

    # construct recognition and decoder networks and initialize them
    recognize, recogn_params = init_mlp(P, [(50, np.tanh), (2*N, gaussian_info)])
    decode,   loglike_params = init_mlp(N, [(50, np.tanh), (2*P, gaussian_mean)])
    loglike = make_loglike(decode)

    # initialize LDS parameters
    pgm_params = init_pgm_param(N)
    params = pgm_params, loglike_params, recogn_params

    # set up encoder/decoder and plotting
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)
    plot = make_plotter(recognize, decode, data, params, prefix=25, plot_every=10)

    # instantiate svae gradient function
    gradfun = make_gradfun(run_inference, recognize, loglike, pgm_prior_params, data)

    # optimize
    params = sgd(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e3, callback=plot),
                 params, num_iters=1000, step_size=1e-3)
