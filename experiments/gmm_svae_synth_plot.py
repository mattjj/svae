from __future__ import division
import numpy as np
import numpy.random as npr
import cPickle as pickle
import matplotlib.pyplot as plt
from itertools import count
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

from gmm_svae_synth import *

# colors = ((1., 0., 0.), (0., 0, 0.), (0., 1., 0.), (1., 1., 0.), (1., 1., 0.),)

colors = np.array([
    (166,206,227), (31,120,180), (178,223,138), (51,160,44),
    (251,154,153), (227,26,28), (253,191,111), (255,127,0),
    (202,178,214), (106,61,154)]) / 256.

colors2 = np.array([
    [ 228, 26, 28 ], [ 55, 126, 184 ], [ 77, 175, 74 ],
    [ 152, 78, 163 ], [ 255, 127, 0 ], [ 255, 255, 51 ],
    [ 166, 86, 40 ], [ 247, 129, 191 ]]) / 256.

colors = np.concatenate((colors, colors2))

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
              gridsize=100, vmin=0., vmax=1., zorder=1)

def decode_density(latent_locations, phi, weight=1.):
    mu, log_sigmasq = decode(latent_locations, phi)
    sigmasq = np.exp(log_sigmasq)

    def density(r):
        distances = np.sqrt(((r[None,:,:] - mu)**2 / sigmasq).sum(2))
        return weight * (norm.pdf(distances) / np.sqrt(sigmasq).prod(2)).mean(0)

    return density

def plot(itr, axs, data, params):
    natparam, phi, psi = params
    ax_data, ax_latent = axs
    K = len(natparam[1])

    def generate_ellipse(mu, Sigma):
        t = np.hstack([np.arange(0, 2*np.pi, 0.01),0])
        circle = np.vstack([np.sin(t), np.cos(t)])
        ellipse = 2. * np.dot(np.linalg.cholesky(Sigma), circle)
        return ellipse[0] + mu[0], ellipse[1] + mu[1]

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

    # plot_or_update(0, ax_data, reconstruction[:,0], reconstruction[:,1],
    #                color='b', marker='x', linestyle='')

    del ax_data.collections[1]  # delete old hexbin artist
    xlim, ylim = ax_data.get_xlim(), ax_data.get_ylim()
    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        ## 
        samples = npr.RandomState(0).multivariate_normal(mu, Sigma, 200)
        density = decode_density(samples, phi, 75. * weight)
        plot_transparent_hexbin(ax_data, density, xlim, ylim, 100, colors[idx % len(colors)])

        ## ellipses
        # x, y = generate_ellipse(mu, Sigma)
        # transformed_x, transformed_y = decode_mean(np.vstack((x, y)).T, phi).T
        # plot_or_update(idx, ax_data, transformed_x, transformed_y,
        #                alpha=min(1., K*weight), linestyle='-', linewidth=1,
        #                zorder=2, color=colors[idx % len(colors)])

    ## make latent space plot

    plot_or_update(0, ax_latent, latent_locations[:,0], latent_locations[:,1],
                   color='k', marker='.', linestyle='', markersize=1)

    for idx, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        x, y = generate_ellipse(mu, Sigma)
        plot_or_update(idx+1, ax_latent, x, y, alpha=min(1., K*weight),
                       linestyle='-', linewidth=2, color=colors[idx % len(colors)])

    ax_latent.relim()
    ax_latent.autoscale_view(True, True, True)


def make_figure():
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(data[:,0], data[:,1], s=1, color='k', marker='.', zorder=2)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].autoscale(False)
    axs[0].axis('off')
    axs[1].axis('off')
    fig.tight_layout()

    return fig, axs


if __name__ == "__main__":
    npr.seed(1)
    data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)

    fig, axs = make_figure()
    with open('gmm_svae_synth_params2.pkl') as infile:
        try:
            for itr in count():
                pickle.load(infile)  # TODO remove me
                pickle.load(infile)  # TODO remove me
                pickle.load(infile)  # TODO remove me
                plot(itr, axs, data, pickle.load(infile))
                filename = 'figures/gmm_{:04d}.png'.format(itr)
                plt.savefig(filename, dpi=150)
                print 'saved {}'.format(filename)
        except EOFError:
            pass
