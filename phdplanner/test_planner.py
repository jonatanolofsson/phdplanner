#!/usr/bin/env python3
# pylint: disable=invalid-name, too-many-locals, no-member, too-many-arguments
"""Test PHD planner, C++/Python interface."""

import argparse
from copy import deepcopy
import logging
import sys
import time
from math import floor
import phdplanner as ppl
import matplotlib.pyplot as plt
import numpy as np
import plot

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger(__name__)
np.random.seed(1)

ARGS = None


def run():
    """Run."""
    nx = 100
    ny = 120
    K = 3
    T = 60
    # N = lambda k: 1000000 if k == 0 else 30000  # noqa
    N = lambda k: 10 if k == 0 else 300  # noqa
    mutation = lambda k: 0.001  # noqa

    params = ppl.Params()
    params.value_factor = 2
    params.straight_reward = 5e-4
    params.diagf_reward = 1e-4
    params.side_reward = 0 #5e-5
    params.max_streak = 10
    params.streak_reward = params.straight_reward
    params.score_power = 3
    params.pD = 0.99
    pMin = 0.5
    unknown = 0.0
    points = np.empty((2, nx * ny))
    i = 0
    for x in range(nx):
        for y in range(ny):
            points[:, i] = np.array([x, y])
            i = i + 1

    phd = np.ones((1, nx * ny)) * unknown
    target_pos = []
    for _ in range(2):
        tpos = np.random.rand(2, 1) * 99
        print("Target at ", tpos.T)
        target_pos.append(tpos)
        phd += (pMin + (1 - pMin) * np.random.rand(1, 1)) \
            * ppl.Gaussian(tpos,
                           np.eye(2) * np.random.rand(1, 1) * 10) \
            .sampled_pos_pdf(points)
    # target_pos = [np.array([[10], [20]])]
    # for tpos in target_pos:
        # phd += ppl.Gaussian(tpos, np.eye(2) * 10).sampled_pos_pdf(points)

    phds = [phd.reshape((nx, ny))]
    agents = [
        np.array([[5], [5]], dtype=np.int32),
        np.array([[5], [50]], dtype=np.int32),
    ]
    if not ARGS.noimages:
        fig = plt.figure(figsize=(20, 20))
        plot.phd(phds[0])
        plt.plot([t[1] for t in target_pos], [t[0] for t in target_pos], 'rx', label='Target centerpoints')
        fig.savefig(f"{ARGS.filename}_phd.png", bbox_inches='tight')

    # evolve_consecutively(phds, agents, params, T, N, mutation, K, target_pos)
    evolve_intertwined(phds, agents, params, T, N, mutation, K, target_pos)


def evolve_intertwined(phds, agents, params, T, N, mutation, K, target_pos):
    """Evolve agents in time lockstep."""
    n = N(0)
    agent_paths = {}
    t = time.time()
    planners = {}
    observed_phds = deepcopy(phds)
    for agent_id, start in enumerate(agents):
        planner = planners[agent_id] = ppl.Planner(observed_phds, params, start, n, T)
        observe(observed_phds, [planner.best_path()], params)
    for planner in planners.values():
        planner.params = params
    print(f"Initialized planners in {time.time()-t} s")
    t = time.time()
    k = 0
    scores = {agent_id: [0] * K for agent_id in planners}
    for agent_id, planner in planners.items():
        scores[agent_id][0] = planner.best_score
    scores["joint"] = [0] * K
    scores["joint"][0] = sum(score[0] for _, score in scores.items())
    best_score = {agent_id: planner.best_score for agent_id, planner in planners.items()}
    best_score["joint"] = sum(score for _, score in best_score.items())
    all_paths = {agent_id: [None] * K for agent_id in planners}

    if not ARGS.noimages:
        fig = plt.figure(figsize=(20, 20))
        plot.phd(phds[0])
        for agent_id, planner in planners.items():
            path = planner.best_path()
            all_paths[agent_id][k] = path
            bweight = planner.w[planner.best_index]
            for i in range(min(n, floor(100 / len(agents)))):
                plot.path(planner.get_path(i * max(1, floor(n / floor(100 / len(agents))))),
                          agent_id, alpha=0.5 * planner.w[i] / bweight)
        for agent_id in planners:
            plot.path(all_paths[agent_id][k], 2 + agent_id, label=f"Agent {agent_id}")
        plt.plot([t[1] for t in target_pos], [t[0] for t in target_pos], 'rx', label='Target centerpoints')
        plt.legend()
        fig.savefig(f"{ARGS.filename}_{k:05}.png", bbox_inches='tight')

    for k in range(1, K):
        if not ARGS.noimages:
            fig.clf()
            plot.phd(phds[0])
        n = N(k)
        for agent_id, planner in planners.items():
            observe(phds, [path for agent, path in agent_paths.items()
                           if agent != agent_id], params)
            planner.update_values(observed_phds)

            planner.evolve(mutation(k), n)
            path = planner.best_path()
            all_paths[agent_id][k] = path
            scores[agent_id][k] = planner.best_score
            scores["joint"][k] += planner.best_score
            if planner.best_score > best_score[agent_id]:
                best_score[agent_id] = planner.best_score
                agent_paths[agent_id] = path
            if not ARGS.noimages:
                bweight = planner.w[planner.best_index]
                for i in range(floor(100 / len(agents))):
                    plot.path(planner.get_path(i * floor(n / floor(100 / len(agents)))),
                              agent_id, alpha=0.5 * planner.w[i] / bweight)

        if scores["joint"][k] > best_score["joint"]:
            best_score["joint"] = planner.best_score

        if not ARGS.noimages:
            for agent_id in planners:
                plot.path(all_paths[agent_id][k], 2 + agent_id, label=f"Agent {agent_id}")
            plt.plot([t[0] for t in target_pos], [t[1] for t in target_pos], 'rx', label='Target centerpoints')
            plt.legend()
            fig.savefig(f"{ARGS.filename}_{k:05}.png", bbox_inches='tight')

        print(f"Finished step {k}")

    if not ARGS.noimages:
        fig = plt.figure(figsize=(20, 20))
        for agent_id in scores:
            plt.plot(scores[agent_id], label=f"Agent {agent_id}")
        plt.legend()
        fig.savefig(f"{ARGS.filename}_scores.png", bbox_inches='tight')

    for agent_id in scores:
        max_value = max(scores[agent_id])
        max_index = scores[agent_id].index(max_value)
        print(f"Found best path for agent {agent_id} at generation {max_index} (score {max_value}).")
    print(f"Agents evolved {K} steps in {time.time()-t} seconds.")


def evolve_consecutively(phds, agents, params, T, N, mutation, K, target_pos):
    """Evolve agents one at a time."""
    agent_paths = {}
    for agent_id, start in enumerate(agents):
        t = time.time()
        n = N(0)
        observed_phds = deepcopy(phds)
        observe(observed_phds, [path for agent, path in agent_paths.items()
                                if agent != agent_id], params)
        planner = ppl.Planner(observed_phds, params, start, n, T)
        print(f"Initialized planner in {time.time()-t} s")
        planner.params = params
        t = time.time()
        fig = plt.figure(figsize=(20, 20))
        scores = []
        best_score = 0
        best_k = 0
        for k in range(1, K):
            if not ARGS.noimages:
                fig.clf()
                plot.phd(phds[0])

            n = N(k)
            planner.evolve(mutation(k), n)
            path = planner.best_path()
            scores.append(planner.best_score)
            if planner.best_score > best_score:
                best_score = planner.best_score
                agent_paths[agent_id] = path
                best_k = k

            if not ARGS.noimages:
                bweight = planner.w[planner.best_index]
                for i in range(100):
                    plot.path(planner.get_path(i * floor(n / 100)),
                              agent_id, alpha=planner.w[i] / bweight)
                plot.path(path, 2 + agent_id)
                plt.plot([t[0] for t in target_pos], [t[1] for t in target_pos], 'rx')
                fig.savefig(f"{ARGS.filename}_{agent_id}_{k:05}.png", bbox_inches='tight')

            print(f"Finished step {k}")

        fig = plt.figure(figsize=(20, 20))
        plt.plot(scores)
        fig.savefig(f"{ARGS.filename}_{agent_id}_scores.png", bbox_inches='tight')
        print(f"Found best path for agent {agent_id} at generation {best_k} (score {best_score}.")
        print(f"Agent {agent_id}'s evolved {K} steps in {time.time()-t} seconds.")


def observe(phds, paths, params):
    """Reduce the phds based on observations."""
    for path in paths:
        for t, position in enumerate(path):
            phds[0 if len(phds) == 1 else t][position[0], position[1]] *= params.pD


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default="planner", nargs="?")
    parser.add_argument('--noimages', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    global ARGS
    ARGS = parse_args(*argv)
    run()


if __name__ == '__main__':
    main(*sys.argv[1:])
