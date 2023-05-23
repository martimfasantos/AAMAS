import random

from . import Agent


class RandomAgent(Agent):
    name = "Random Agent"

    def step(self, obs):
        return random.choice([i for i in range(6) ])
