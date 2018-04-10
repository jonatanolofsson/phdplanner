"""Create PHD plot."""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt


np.random.seed(1)


def draw(args):
    """Create plot."""
    for phdfile in args.phdfiles:
        fig = plt.figure(figsize=(30, 30))
        ax = fig.gca(projection='3d')
        with open(phdfile, 'r') as fhandle:
            phddata = json.load(fhandle)
            points = np.array(phddata['points'])
            phd = np.array(phddata['phd'])
            xs = points[0, :]
            ys = points[1, :]
            zs = phd
            xi = np.linspace(min(xs), max(xs), 100)
            yi = np.linspace(min(ys), max(ys), 100)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata(xs, ys, zs, xi, yi, interp='linear')
            ax.plot_surface(X, Y, Z)

        if args.show:
            plt.show()
        else:
            plt.gcf().savefig(os.path.splitext(os.path.basename(phdfile))[0],
                              bbox_inches='tight')


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phdfiles", nargs='+')
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw(args)


if __name__ == '__main__':
    main(*sys.argv[1:])
