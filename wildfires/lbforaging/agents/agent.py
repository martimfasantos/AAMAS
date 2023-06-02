import logging

import numpy as np

TILES_PER_FIRE  =4
_MAX_INT = 999999


class Agent:
    name = "Prototype Agent"

    def __repr__(self):
        return self.name

    def __init__(self, player):
        self.logger = logging.getLogger(__name__)
        self.player = player

    def __getattr__(self, item):
        if item == "water":
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
            min_idx = (abs(r - y) + abs(c - x)).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]
    
    def _strongest_front_closest_fire(self, obs, max_fire_level=None, start=None):

        def fire_fronts(field, y, x):
            possible_positions = [[y-1, x-1], [y-1, x], [y-1, x+1], [y, x-1], [y, x],
                                [y, x+1], [y+1, x-1], [y+1, x], [y+1, x+1]] # Assuming 9 possible positions
            active_fronts = [front for front in possible_positions if y >= 0 and y < field.shape[0] 
                               and x >= 0 and x < field.shape[1] and field[front[0], front[1]] > 0]
            return [coord[0] for coord in active_fronts], [coord[1] for coord in active_fronts]

        if start is None:
            y,x = self.observed_position
        else:
            y,x = start

        field = np.copy(obs.field)

        if max_fire_level:
            field[field > max_fire_level] = 0

        r, c = np.where(field > 0)
        try:
            min_dist = (abs(r - y) + abs(c - x)).argmin()
            # check if there are adjacent in all directions fires with more intensity
            r_adj_fronts, c_adj_fronts = fire_fronts(field, r[min_dist], c[min_dist])
            max_idx = np.argmax(field[r_adj_fronts, c_adj_fronts])

        except ValueError:
            return None
        
        # if more than one adjacent fire with more intensity, choose the closest one
        if max_idx.size > 1:
            min_dist = (abs(r_adj_fronts[max_idx] - y) + abs(c_adj_fronts[max_idx] - x)).argmin()
            max_idx = max_idx[min_dist]

        return r_adj_fronts[max_idx], c_adj_fronts[max_idx]
    
    def _closest_water_source(self, obs):

        y,x = self.observed_position
        field = np.copy(obs.field)

        r, c = np.where(field == -1)
        try:
            min_idx = (abs(r - y) + abs(c - x)).argmin()
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

    
   
       
