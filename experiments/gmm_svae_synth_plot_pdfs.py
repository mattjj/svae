from __future__ import division
import numpy as np
import numpy.random as npr
import cPickle as pickle
import matplotlib.pyplot as plt
from itertools import count
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from operator import itemgetter
from functools import partial

from gmm_svae_synth import decode as gmm_decode, make_pinwheel_data, normalize, \
    dirichlet, niw, encode_mean, decode_mean
from svae.forward_models import mlp_decode
mlp_decode = partial(mlp_decode, tanh_scale=1000., sigmoid_output=False)

gridsize=75
num_samples=2000

colors = np.array([
[106,61,154],  # Dark colors
[31,120,180],
[51,160,44],
[227,26,28],
[255,127,0],
[166,206,227],  # Light colors
[178,223,138],
[251,154,153],
[253,191,111],
[202,178,214],
]) / 256.0


def get_hexbin_coords(ax, xlims, ylims, gridsize):
    coords = ax.hexbin([], [], gridsize=gridsize, extent=tuple(xlims)+tuple(ylims)).get_offsets()
    del ax.collections[-1]
    return coords

def plot_transparent_hexbin(ax, func, xlims, ylims, gridsize, color):
    cdict = {'red':   ((0., color[0], color[0]), (1., color[0], color[0])),
             'green': ((0., color[1], color[1]), (1., color[1], color[1])),
             'blue':  ((0., color[2], color[2]), (1., color[2], color[2])),
             'alpha': ((0., 0., 0.), (1., 1., 1.))}

    new_cmap = LinearSegmentedColormap('Custom', cdict)
    plt.register_cmap(cmap=new_cmap)

    coords = get_hexbin_coords(ax, xlims, ylims, gridsize)
    c = func(coords)
    x, y = coords.T

    ax.hexbin(x.ravel(), y.ravel(), c.ravel(),
              cmap=new_cmap, linewidths=0., edgecolors='none',
              gridsize=gridsize, vmin=0., vmax=1., zorder=1)

def decode_density(latent_locations, phi, decode, weight=1.):
    mu, log_sigmasq = decode(latent_locations, phi)
    sigmasq = np.exp(log_sigmasq)

    def density(r):
        distances = np.sqrt(((r[None,:,:] - mu)**2 / sigmasq).sum(2))
        return weight * (norm.pdf(distances) / np.sqrt(sigmasq).prod(2)).mean(0)

    return density

def set_border_around_data(ax, data, border=0.1):
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    ax.set_xlim([xmin - (xmax - xmin) * border,
                 xmax + (xmax - xmin) * border])
    ax.set_ylim([ymin - (ymax - ymin) * border,
                 ymax + (ymax - ymin) * border])

def generate_ellipse(mu, Sigma):
    t = np.hstack([np.arange(0, 2*np.pi, 0.01),0])
    circle = np.vstack([np.sin(t), np.cos(t)])
    ellipse = 2. * np.dot(np.linalg.cholesky(Sigma), circle)
    return ellipse[0] + mu[0], ellipse[1] + mu[1]

def plot(axs, data, params):
    natparam, phi, psi = params
    ax_data, ax_latent = axs
    K = len(natparam[1])

    def plot_or_update(idx, ax, x, y, alpha=1, **kwargs):
        if len(ax.lines) > idx:
            ax.lines[idx].set_data((x, y))
            ax.lines[idx].set_alpha(alpha)
        else:
            ax.plot(x, y, alpha=alpha, **kwargs)

    dir_hypers, all_niw_hypers = natparam
    weights = normalize(np.exp(dirichlet.expectedstats(dir_hypers)))
    components = map(niw.expected_standard_params, all_niw_hypers)

    latent_locations = encode_mean(data, natparam, psi)
    reconstruction = decode_mean(latent_locations, phi)

    ## make data-space plot

    ax_data.scatter(data[:, 0], data[:, 1], s=1, color='k', marker='.', zorder=2)
    ax_data.collections[1:] = []
    set_border_around_data(ax_data, data)


    xlim, ylim = ax_data.get_xlim(), ax_data.get_ylim()
    for idx, (weight, (mu, Sigma)) in enumerate(
            sorted(zip(weights, components), key=itemgetter(0))):
        samples = npr.RandomState(0).multivariate_normal(mu, Sigma, num_samples)
        density = decode_density(samples, phi, gmm_decode, 75. * weight)
        plot_transparent_hexbin(ax_data, density, xlim, ylim, gridsize, colors[idx % len(colors)])


    ## make latent space plot

    plot_or_update(0, ax_latent, latent_locations[:,0], latent_locations[:,1],
                   color='k', marker='.', linestyle='', markersize=1)
    set_border_around_data(ax_latent, latent_locations)

    for idx, (weight, (mu, Sigma)) in enumerate(
            sorted(zip(weights, components), key=itemgetter(0))):
        x, y = generate_ellipse(mu, Sigma)
        plot_or_update(idx+1, ax_latent, x, y, alpha=min(1., K*weight),
                       linestyle='-', linewidth=2, color=colors[idx % len(colors)])

    ax_latent.relim()
    ax_latent.autoscale_view(True, True, True)

def make_figure():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return fig, ax

def save_figure(fig, filename):
    fig.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
    print 'saved {}'.format(filename)

def plot_data(data):
    fig, ax = make_figure()
    ax.plot(data[:,0], data[:,1], 'k.')
    save_figure(fig, 'figures/mainfig_mix_data.png')
    plt.close()

def plot_gmm(filename, data):
    def load_gmm_params(filename):
        with open(filename) as f:
            params = pickle.load(f)
        return params

    params = load_gmm_params(filename)
    fig, ax = make_figure()
    ax.plot(data[:,0], data[:,1], 'k.')
    for idx, (weight, mu, Sigma) in enumerate(sorted(params, key=itemgetter(0))):
        x, y = generate_ellipse(mu, Sigma)
        ax.plot(x, y, alpha=min(1., len(params)*weight/2),
                linestyle='-', linewidth=2, color=colors[idx % len(colors)])
    save_figure(fig, 'figures/mainfig_mix_plaingmm.png')


def plot_vae_density(filename, data):
    def load_vae_density_params(filename):
        with open(filename) as f:
            _, phi = pickle.load(f)
        return phi

    phi = load_vae_density_params(filename)
    fig, ax = make_figure()
    data = 1./10 * data  # density network was fit to unscaled data
    ax.plot(data[:,0], data[:,1], 'k.')
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    samples = npr.RandomState(0).randn(2*num_samples, 2)
    density = decode_density(samples, phi, mlp_decode, weight=0.6)
    plot_transparent_hexbin(ax, density, xlim, ylim, gridsize, colors[0])
    save_figure(fig, 'figures/mainfig_mix_densitynetwork.png')

def plot_gmm_svae(filename, data):
    def load_gmm_svae_params(filename):
        with open(filename) as f:
            try:
                for _ in range(20000): gmm_svae_params = pickle.load(f)
            except EOFError: pass
            else: print 'did not finish loading {}'.format(filename)
        return gmm_svae_params

    gmm_svae_params = load_gmm_svae_params(filename)
    fig1, ax1 = make_figure()
    fig2, ax2 = make_figure()
    plot((ax1, ax2), data, gmm_svae_params)
    save_figure(fig1, 'figures/mainfig_mix_observed.png')
    save_figure(fig2, 'figures/mainfig_mix_latent.png')
    plt.close('all')


if __name__ == "__main__":
    npr.seed(1)
    data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)
    plot_data(data)
    plot_gmm('pinwheel_gmm.pkl', data)
    plot_vae_density('warped_mixture_density_network.pkl', data)
    plot_gmm_svae('gmm_svae_synth_params.pkl', data)
