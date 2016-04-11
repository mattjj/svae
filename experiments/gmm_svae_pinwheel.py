from __future__ import division
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt


def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum('ti,tij->tj', features, rotations)


if __name__ == "__main__":
    features = make_pinwheel(0.3, 0.1, 5, 250, 0.3)
    plt.plot(features[:,0], features[:,1], 'b.')
    plt.show()
