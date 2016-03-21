from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.signal import sawtooth


triangle = lambda t: sawtooth(np.pi*t, width=0.5)
make_dot_trajectory = lambda x0, v: lambda t: triangle(v*(t + (1+x0)/2.))
make_renderer = lambda grid, sigma: lambda x: np.exp(-1./2 * (x - grid)**2/sigma**2)

def make_dot_data(image_width, T, num_steps, x0=0.0, v=0.5, render_sigma=0.2, noise_sigma=0.1):
    grid = np.linspace(-1, 1, image_width, endpoint=True)
    render = make_renderer(grid, render_sigma)
    x = make_dot_trajectory(x0, v)
    images = np.vstack([render(x(t)) for t in np.linspace(0, T, num_steps)])
    return images + noise_sigma * npr.randn(*images.shape)
