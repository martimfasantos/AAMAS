import logging

import numpy as np

_MAX_INT = 999999


class Agent:
    name = "Prototype Agent"

    def __repr__(self):
        return self.name

    def __init__(self, player):
        self.logger = logging.getLogger(__name__)
        self.player = player

    def __getattr__(self, item):
        if(item == "water"):
            return self.controller.water
        return getattr(self.player, item)

    def _step(self, obs):
        self.observed_position = next(
            (x for x in obs.players if x.is_self), None
        ).position

        # saves the action to the history
        action = self.step(obs)
        self.history.append(action)

        return action

    def step(self, obs):
        raise NotImplemented("You must implement an agent")

    def _closest_fire(self, obs, max_fire_level=None, start=None):

        if start is None:
            y,x = self.observed_position
        else:
            y,x = start

        field = np.copy(obs.field)

        if max_fire_level:
            field[field > max_fire_level] = 0

        r, c = np.where(field > 0)
        try:
            min_idx = (abs(r - y)  + abs(c - x)).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]
    
    def _closest_water_source(self, obs):

        y,x = self.observed_position
        field = np.copy(obs.field)

       
        r, c = np.where(field == -1)
        try:
            min_idx = (abs(r - y) + abs(c - x) ).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]

    def _make_state(self, obs):

        state = str(obs.field)
        for c in ["]", "[", " ", "\n"]:
            state = state.replace(c, "")

        for a in obs.players:
            state = state + str(a.position[0]) + str(a.position[1]) + str(a.level)

        return int(state)

    def cleanup(self):
        pass

    
   
       
