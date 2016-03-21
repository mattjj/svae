from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def make_grid(grid_sidelen, imagevecs):
    im_sidelen = int(np.sqrt(imagevecs.shape[1]))
    reshaped = imagevecs.reshape(grid_sidelen,grid_sidelen,im_sidelen,im_sidelen)
    return np.vstack([np.hstack([
        img.reshape(im_sidelen,im_sidelen)
        for img in col]) for col in reshaped])


def plot_random_examples(x, save=True):
    fig = plt.figure()
    x = x if x.ndim == 2 else np.reshape(x, (x.shape[0], -1))
    subset = x[np.random.choice(x.shape[0], 25)]
    plt.imshow(make_grid(5, subset), cmap='gray')
    if save:
        plt.savefig('random_examples.png')
        plt.close(fig)


def plot_samples(itr, samplefun, save=True):
    fig = plt.figure()
    plt.imshow(make_grid(5, samplefun(25)), cmap='gray')
    if save:
        plt.savefig('samples_%03d.png' % itr)
        print 'saved samples_%03d.png' % itr
        plt.close(fig)
