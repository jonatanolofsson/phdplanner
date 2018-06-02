"""Plot helper stuff."""
# pylint: disable=invalid-name
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState


CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256 * 256, 3))


def phd3d(phd_, *args, **kwargs):
    """Plot 3d PHD."""
    fig = plt.figure(figsize=(30, 30))
    ax = fig.gca(projection='3d')
    xi = np.arange(phd_.shape[1])
    yi = np.arange(phd_.shape[0])
    X, Y = np.meshgrid(xi, yi)
    ax.plot_surface(X, Y, phd_, *args, **kwargs)


def path3d(path_, phd_, c=0, *args, **kwargs):
    """Plot 3d path."""
    zpath = [phd_[x, y] for x, y in path_.T]
    plt.gca().plot(path_[0, :], path_[1, :], zpath, color=CMAP(c), *args, **kwargs)


def phd(phd_, *args, **kwargs):
    """Plot."""
    img = plt.gca().imshow(phd_, *args, origin='lower', vmin=0, vmax=phd_.max(), **kwargs)
    plt.colorbar(img)


def path(path_, c=0, *args, **kwargs):
    ax = plt.gca()
    ax.plot(path_[1, :], path_[0, :], color=CMAP(c), *args, **kwargs)
