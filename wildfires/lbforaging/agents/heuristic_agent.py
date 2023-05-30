import random
import numpy as np
from . import Agent
from ..foraging.environment import Action
from ..foraging import FireTruck


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))
    
    def _compute_turn(self, direction, targetedDirection):
        turns = [Action.NONE, Action.TURN_RIGHT, Action.TURN_AROUND, Action.TURN_LEFT]
        return turns[(targetedDirection.value - direction.value) % 4]

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target
        
        if r < y:
            if Action.NORTH in allowed:
                return Action.NORTH
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.NORTH)
        
        if r > y:
            if Action.SOUTH in allowed:
                return Action.SOUTH
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.SOUTH)
        
        if c > x :
            if(Action.EAST in allowed):
                return Action.EAST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.EAST)
            return Action.EAST
        
        if c < x:
            if(Action.WEST in allowed):
                return Action.WEST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.WEST)
            return Action.WEST
        
        # if we reach here, no action is possible to move towards target (choose one randomly)
        raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
	H1 agent always goes to the closest fire
	"""

    name = "H1"

    def step(self, obs):
        try:
            r, c = self._closest_fire(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.EXTINGUISH

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)
        
class H1_1(HeuristicAgent):
    """
	H1 agent always goes to the strongest front of the closest fire
	"""

    name = "H1_1"

    def step(self, obs):
        try:
            r, c = self._closest_fire(obs)
        except TypeError:
            return random.choice(obs.actions)
        x, y = self.observed_position

        if (abs(r - x) + abs(c - y)) == 1:
            return Action.EXTINGUISH

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""

    name = "H2"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._closest_fire(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.EXTINGUISH

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
	H3 Agent always goes to the closest food with compatible level
	"""

    name = "H3"

    def step(self, obs):

        try:
            r, c = self._closest_fire(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.EXTINGUISH

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
	H4 Agent goes to the one visible food which is closest to all visible players
	 such that the sum of their and H4's level is sufficient to load the food
	"""

    name = "H4"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        try:
            r, c = self._closest_fire(obs, players_sum_level, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.EXTINGUISH

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)
