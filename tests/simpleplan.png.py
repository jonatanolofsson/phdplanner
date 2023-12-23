#!/usr/bin/env python3
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, too-many-locals
"""Track ice from Norut flights."""
import os
import sys
import inspect
from copy import deepcopy
from math import floor
import argparse
import asyncio
import logging
import time as pytime
import numpy as np
from asyncinit import asyncinit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lmb
from aniceday import gprdata


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHD_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PHD_DIR)

import phdplanner as ppl
import lmb


@asyncinit
class PhdPlot(ppl.Application, lmb.Application):
    """Main application."""

    async def __init__(self):
        """Init."""
        ppl.Application.__init__(self)
        lmb.Application.__init__(self)

        # Tracker settings
        self.model = lmb.CV(0.5, 1e0)
        self.report_covariance = np.eye(2) * 1000
        self.origin = np.array([50, 10])

        self.sensor = lmb.PositionSensor()
        self.sensor.pD = 0.5
        self.sensor.kappa = 1 / (1000 * 1000)
        self.sensor.pv = self.sensor.pv * (200 * 200)
        self.region = lmb.AABBox(-200, -200, 200, 200).llaabbox(self.origin)
        self.sensor.fov = lmb.BBox(self.region)

        self.unknown = 0.02

        self.nof_scans = 3
        self.tparams = lmb.Params()
        self.tparams.r_lim = 0.1
        self.tparams.w_lim = 0.01;
        self.tparams.nhyp_max = 10;
        self.tparams.rB_max = 0.8;
        self.tparams.nstd = 1.9;
        self.tparams.cw_lim = 0.01;
        self.tparams.cov_lim = 400000;
        self.tparams.tscale = 3600;


        self.ppl_nofsteps = 600
        self.ppl_population_size = lambda k: 1000
        self.ppl_agents = [
            np.array([[10], [5]], dtype=np.int32),
            # np.array([[5], [30]], dtype=np.int32),
        ]
        self.pplparams.value_factor = 2
        self.pplparams.straight_reward = 0  # 5e-4
        self.pplparams.diagf_reward = 0  # 1e-4
        self.pplparams.side_reward = 0 #5e-5
        self.pplparams.max_streak = 10
        self.pplparams.streak_reward = 0
        self.pplparams.score_power = 3
        self.pplparams.pD = 0.99

    async def lambdaB(self, reports, fov):
        """lambdaB callback."""
        return max(
            0.8 * (1.5 * len(reports) - (await self.enof_targets(fov.aabbox()))),
            0.05 * len(reports))

    def plan(self, phd):
        """Make plans."""
        observed_phd = deepcopy(phd)
        neregion = self.region.neaabbox(self.origin)
        gridsize = np.array([
            (neregion.max[0] - neregion.min[0]) / self.phdsize[0],
            (neregion.max[1] - neregion.min[1]) / self.phdsize[1]])
        corner = (neregion.min[0], neregion.min[1])

        paths = {}
        for agent_id, nestart in enumerate(self.ppl_agents):
            start = np.floor((np.squeeze(nestart) - corner) / gridsize)
            n = self.ppl_population_size(0)
            planner = self.planners[agent_id] = \
                ppl.Planner(observed_phd, self.pplparams, start, self.ppl_nofsteps, n)
            self.observe(observed_phd, [planner.best_path()], self.pplparams)
            paths[agent_id] = []
            bweight = planner.w[planner.best_index]
            for i in range(min(n, floor(100 / len(self.ppl_agents)))):
                which = i * max(1, floor(n / floor(100 / len(self.ppl_agents))))
                path = self.decodepath(planner.get_path(which), corner, gridsize)
                weight = planner.w[i]

                ppl.plot.path(path, agent_id, alpha=0.4 * weight / bweight)
            path = self.decodepath(planner.best_path(), corner, gridsize)
            paths[agent_id].append((bweight, path))
            ppl.plot.path(path, agent_id + 100)
            return paths

    async def scans(self):
        """Get scans."""
        reports = [
            lmb.GaussianReport(self.sensor,
                               lmb.cf.ne2ll(np.random.rand(2, 1) * 99, self.origin),
                               np.eye(2) * 40)
            for _ in range(10)]
        lamB = self.lambdaB(reports, self.sensor.fov)
        if inspect.isawaitable(lamB):
            lamB = await lamB
        self.sensor.lambdaB = lamB

        yield 0, self.sensor, reports

    async def run(self):
        """Run application."""
        LOGGER.debug("Starting application.")
        self.tracker = lmb.Tracker(self.tparams)

        plt.figure("phd", figsize=(30, 30))
        last_time = None
        k = -1
        async for time, sensor, reports in self.scans():
            k += 1
            if last_time is None:
                last_time = time

            if self.origin is None:
                if sensor:
                    self.origin = sensor.fov.corners.mean(axis=1)
                else:
                    continue
            if self.region is None:
                if sensor:
                    self.region = sensor.fov.aabbox()
                else:
                    continue

            # Run the LMB filter
            if time != last_time and self.model:
                await self.predict(self.model, time, last_time)
            if sensor:
                # reports = reports[:3]
                reports = await self.correct(sensor, reports, time)

            tstamp = pytime.strftime("%Y%m%d%H%M", pytime.localtime(time))

            # Get tracker statistics
            # pre_enof_targets = await self.enof_targets()

            # Sample and prepare PHD
            neregion = self.region.neaabbox(self.origin)
            extent = (neregion.min[1], neregion.max[1], neregion.min[0], neregion.max[0])
            phd = await self.sample_phd()
            plt.figure("phd")
            plt.clf()
            lmb.plot.scan(reports, self.origin)
            lmb.plot.phd(phd, extent=extent)
            self.plan(phd)
            plt.gcf().savefig(os.path.splitext(__file__)[0], bbox_inches='tight')
            last_time = time


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)


async def main(*argv):
    """Main."""
    LOGGER.debug("Entering main.")
    args = parse_args(*argv)
    app = await PhdPlot()
    await app.run()


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main(*sys.argv[1:]))

