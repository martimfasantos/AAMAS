#!/usr/bin/env python3
import argparse
import logging
import time
import gym
import numpy as np
import warnings
from gym.envs.registration import register
from utils import *
from lbforaging.foraging.environment import TILES_PER_FIRE



SLEEP_TIME = 0.5

logger = logging.getLogger(__name__)
logger.propagate = False

warnings.filterwarnings("ignore")
       

def _game_loop(env, render, debug, team):
    """
    """
    obs,_ = env.reset(team=team)
    done = False

    if render:
        
        env.render()
        if debug:
            input()
        else:
            time.sleep(SLEEP_TIME)

    while not done:

        actions = [player.step(obs[i]) for i, player in enumerate(env.players)]

        obs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            pass

        if render:
            env.render()
            if debug:
                input()
            else:
                time.sleep(SLEEP_TIME)

        done = np.all(ndone)
logger = logging.getLogger(__name__)

def main(game_count, render, fires, steps_incr, n_agents, compare, mode, debug, max_steps, size=16, c=False):
    
    if compare:
        teams = generateTeams(mode, n_agents, compare=True)
    else:
        teams = generateTeams(mode, n_agents)

    results = {}

    for name, team in teams.items():

        print(f"Running with team: {name}")
        # compute the size of the team
        agents = sum([len(team[vehicle]) for vehicle in team])
        
        register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(size, agents, TILES_PER_FIRE*fires, "-coop" if c else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": agents,
            "field_size": (size, size),
            "max_fires": TILES_PER_FIRE*fires,
            "steps_incr": steps_incr,
            "sight": size,
            "max_episode_steps": max_steps,
            "force_coop": c,
        },
        )
        env = gym.make(f"Foraging-{size}x{size}-{agents}p-{TILES_PER_FIRE*fires}f-v2")

        results[name] = np.zeros(game_count)

        for episode in range(game_count):
            _game_loop(env, render, debug, team)
            print(f"Episode {episode+1} of {game_count} finished with {env.current_step} steps.")
            results[name][episode] = env.current_step

    if compare:
        # Compare results
        compare_results(
            results,
            title="Teams Comparison on 'Wildfires' Environment",
            colors=["orange", "green", "blue", "gray"][:len(results)]
        )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    parser.add_argument(
        "--max_steps", type=int, default=400, help="How many steps in each episode"
    )
    parser.add_argument("--fires", type=int, default=3, help="How many fires to start with")
    parser.add_argument("--steps_incr", type=int, default=None, help="How many steps to increase the fire level by one")

    parser.add_argument("--n_agents", type=int, default=2, help="How many agents to run with")
    parser.add_argument("--compare", default=False, action="store_true", help="Plot graphs to compare teams")
    parser.add_argument("--mode", type=int, default=0, help="How should agents behave:\n\t\
                        DEFAULT:\n\t\
                        0 - Randomly\n\t\
                        1 - Pseudo-randomly\n\t\
                        2 - Greedy Heuristic 1\n\t\
                        3 - Greedy Heuristic 2\n\t\
                        4 - Greedy Heuristic 3\n\t\
                        5 - Greedy Heuristic 4\n\t\
                        6 - Greedy Heuristic 5\n\t\
                        7 - Social Conventions 1\n\t\
                        8 - Social Conventions 2\n\t\
                        9 - Social Conventions 3\n\t\
                        10 - Role Based 1\n\t\
                        11 - Role Based 2\n\t\
                        12 - Self defined teams\n\t\
                        COMPARISON MODE:\n\t\
                        1 - Random vs Pseudo-random\n\t\
                        2 - Greedy Heuristics\n\t\
                        3 - Social Conventions\n\t\
                        4 - Role Based"
                        )

    args = parser.parse_args()
    main(args.times, args.render, args.fires, args.steps_incr, args.n_agents, args.compare, args.mode, args.debug, args.max_steps)

