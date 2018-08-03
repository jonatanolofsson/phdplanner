"""Phdplanner application."""
import numpy as np
from . import plot
import phdplanner as ppl

class Application:
    def __init__(self):
        """Init."""
        self.ppl_nofsteps = 600
        self.ppl_population_size = lambda k: 20 if k == 0 else 30000  # noqa
        self.ppl_agents = []
        self.pplparams = ppl.Params()
        self.planners = {}

    def decodepath(self, path, corner, gridsize):
        """Convert ppl path to world coordinates."""
        dpath = np.empty(path.shape)
        for t in range(path.shape[1]):
            dpath[:, t] = (path[:, t] * gridsize + corner).astype(np.int)
        return dpath


    def observe(self, phd, paths, params):
        """Reduce the phd based on observations."""
        for path in paths:
            for t in range(path.shape[1]):
                phd[path[0, t], path[1, t]] *= 1 - params.pD
