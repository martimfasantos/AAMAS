import numpy as np

from . import Agent

class PseudoRandomAgent(Agent):
    name = "Pseudo  Random Agent"

    # agent that prefers to move instead of turning or doing nothing
    def step(self, obs):
        from lbforaging.foraging.environment import Action
    
        weights = np.ones(len(obs.actions))
        for i, action in enumerate(obs.actions):
            if action == Action.NONE:
                weights[i] = 0.1
            elif action == Action.TURN_LEFT or action == Action.TURN_RIGHT \
            or action == Action.TURN_AROUND:
                weights[i] = 0.3
        return np.random.choice(obs.actions, p=weights/np.sum(weights))
