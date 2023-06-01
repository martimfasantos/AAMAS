#!/usr/bin/env python3
import argparse
import logging
import time
import gym
import numpy as np
from lbforaging.foraging.environment import TILES_PER_FIRE
from lbforaging.agents import *
import warnings
from gym.envs.registration import register


SLEEP_TIME = 0.5

logger = logging.getLogger(__name__)
logger.propagate = False

warnings.filterwarnings("ignore")

def generateTeams(mode, n_agents):
    """
    Generates a dictionary of teams and their agents
    """

    if(mode == 0):
        return {
            "Random Agents": {
                "Helicopters": [RandomAgent for _ in range(n_agents // 2)],
                "Firetrucks": [RandomAgent for _ in range(n_agents // 2)]
            }
        }
    elif(mode == 1):
        return {
            "PseudoRandom Agents": {
                "Helicopters": [PseudoRandomAgent for _ in range(n_agents // 2)],
                "Firetrucks": [PseudoRandomAgent for _ in range(n_agents // 2)]
            }
        }
    elif(mode == 2):
        return {
            "Greedy H1 Agents": {
                "Helicopters": [H1 for _ in range(n_agents // 2)],
                "Firetrucks": [H1 for _ in range(n_agents // 2)]
            }
        }
    elif(mode == 3):
        return {
            "Greedy H2 Agents": {
                "Helicopters": [H2 for _ in range(n_agents // 2)],
                "Firetrucks": [H2 for _ in range(n_agents // 2)]
            }
        }
    elif(mode == 4):
        return {
            "Greedy H3 Agents": {
                "Helicopters": [H3 for _ in range(n_agents // 2)],
                "Firetrucks": [H3 for _ in range(n_agents // 2)]
            }
        }
    elif(mode == 5):
        return {
            "Greedy H4 Agents": {
                "Helicopters": [H4 for _ in range(n_agents // 2)],
                "Firetrucks": [H4 for _ in range(n_agents // 2)]
            }
        }
    
    else:
        return {
            "Random Agents": {
                "Helicopters": [RandomAgent, 
                            ],
                "Firetrucks": [RandomAgent,
                            ]
            },
            "Heuristic Agents": {
                "Helicopters": [H1,
                                ],
                "Firetrucks": [H1,
                                ]
            }

        }
   
    

def _game_loop(env, render, debug, team):
    """
    """
    obs,_ = env.reset(team=team)
    done = False

    if render:
        
        env.render()
        if(debug):
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
            if(debug):
                input()
            else:
                time.sleep(SLEEP_TIME)

        done = np.all(ndone)
logger = logging.getLogger(__name__)

def main(game_count, render, fires, n_agents, mode, debug, max_steps,size=16,c=False):
    teams = generateTeams(mode, n_agents)

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
            "sight": size,
            "max_episode_steps": max_steps,
            "force_coop": c,
        },
        )
        env = gym.make(f"Foraging-{size}x{size}-{agents}p-{TILES_PER_FIRE*fires}f-v2")

        for episode in range(game_count):
            _game_loop(env, render, debug, team)
            print(f"Episode {episode+1} of {game_count} finished with {env.current_step} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render",default=False, action="store_true")
    parser.add_argument("--debug",default=False, action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    parser.add_argument(
        "--max_steps", type=int, default=400, help="How many steps in each episode"
    )
    parser.add_argument("--fires", type=int, default=3, help="How many fires to start with")

    parser.add_argument("--n_agents", type=int, default=2, help="How many agents to run with")
    parser.add_argument("--mode", type=int, default=0, help="How should agents behave:\n\t0 - Randomly\n\t1 - \
                        Pseudo-randomly\n\t2 - Greedy Heuristic 1\n\t3 - Greedy Heuristic2\n\t4 - Self defined teams")

    args = parser.parse_args()
    main(args.times, args.render, args.fires, args.n_agents, args.mode, args.debug,args.max_steps)

