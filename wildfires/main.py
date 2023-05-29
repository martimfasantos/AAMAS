#!/usr/bin/env python3
import argparse
import logging
import random
import time
import gym
import numpy as np
from lbforaging.foraging.environment import TILES_PER_FIRE
import warnings
from gym.envs.registration import register

SLEEP_TIME = 0.5


logger = logging.getLogger(__name__)
logger.propagate = False

warnings.filterwarnings("ignore")


def _game_loop(env, render,debug):
    """
    """
    obs,info = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(SLEEP_TIME)

    while not done:

        
        actions = [player.step(obs[i]) for i,player in enumerate(env.players)]
        
        obs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            pass

        if render:
            env.render()
            if(debug):
                input()
            else:
                time.sleep(SLEEP_TIME)

        done = np.all(ndone)
logger = logging.getLogger(__name__)

def main(game_count, render, fires, agents, debug, size=16,c=False,):
    register(
    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(size, agents, TILES_PER_FIRE*fires, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": agents,
        "max_player_level": 3,
        "field_size": (size, size),
        "max_food": TILES_PER_FIRE*fires,
        "sight": size,
        "max_episode_steps": 50,
        "force_coop": c,
    },
)
    env = gym.make(f"Foraging-16x16-{agents}p-{TILES_PER_FIRE*fires}f-v2")
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render,debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render",default=False, action="store_true")
    parser.add_argument("--debug",default=False, action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    parser.add_argument("--fires", type=int, default=3, help="How many fires to start with")
    parser.add_argument("--agents", type=int, default=2, help="How many agents to run with")


    args = parser.parse_args()
    main(args.times, args.render,args.fires,args.agents,args.debug)

