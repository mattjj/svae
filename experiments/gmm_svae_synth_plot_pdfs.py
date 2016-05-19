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

from gmm_svae_synth import *

gridsize=75
num_samples=200

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

def decode_density(latent_locations, phi, weight=1.):
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

    ax_data.scatter(data[:, 0], data[:, 1], s=1, color='k', marker='.', zorder=2)
    ax_data.collections[1:] = []
    set_border_around_data(ax_data, data)


    xlim, ylim = ax_data.get_xlim(), ax_data.get_ylim()
    for idx, (weight, (mu, Sigma)) in enumerate(
            sorted(zip(weights, components), key=itemgetter(0))):
        samples = npr.RandomState(0).multivariate_normal(mu, Sigma, num_samples)
        density = decode_density(samples, phi, 75. * weight)
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
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    #ax2.set_aspect('equal')
    #ax1.set_aspect('equal')
    #ax1.autoscale(False)
    #ax2.autoscale(False)
    ax1.axis('off')
    ax2.axis('off')
    fig1.tight_layout()
    fig2.tight_layout()

    return fig1, fig2, ax1, ax2


if __name__ == "__main__":
    npr.seed(1)
    data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)

    fig1, fig2, ax1, ax2 = make_figure()
    with open('gmm_svae_synth_params.pkl') as infile:
        for itr in range(500):
            try:
                pickle.load(infile)
                print "."
            except EOFError:
                pass

        print "Plotting..."
        plot(itr, (ax1, ax2), data, pickle.load(infile))

        print "Saving..."
        filename = 'figures/mainfig_mix_observed.png'
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig1.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
        print 'saved {}'.format(filename)

        filename = 'figures/mainfig_mix_latent.png'
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig2.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
        print 'saved {}'.format(filename)

    print "Done"