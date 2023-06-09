import random
import numpy as np
from . import Agent
from ..wildfires_env import Action, ExtinguishingMode
from ..wildfires_env import FireTruck


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def __init__(self, player):
        super().__init__(player)
        self.target_fire = None
        self.target_water_resource = None

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))
    
    def _compute_turn(self, direction, targeted_direction):
        turns = [Action.NONE, Action.TURN_RIGHT, Action.TURN_AROUND, Action.TURN_LEFT]
        return turns[(targeted_direction.value - direction.value) % 4]

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        # self.forbidden prevents the agent from going back and forth between two tiles
        if r < y and Action.NORTH:
            if Action.NORTH in allowed:
                return Action.NORTH
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.NORTH)
        
        if r > y and Action.SOUTH:
            if Action.SOUTH in allowed:
                return Action.SOUTH
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.SOUTH)
            
        if c > x and Action.EAST:
            if (Action.EAST in allowed):
                return Action.EAST
            elif (isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.EAST)
            
        if c < x and Action.WEST:
            if(Action.WEST in allowed):
                return Action.WEST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.WEST)
            
        return Action.NONE
            

    def _refill_water(self, obs):
        
        y,x = self.observed_position
        if self.target_water_resource is not None:
            r,c = self.target_water_resource
        else:
            r,c = self._closest_water_source(obs)
            self.target_fire = (r,c)

        if (abs(r - y) + abs(c - x)) == 1:
            self.target_fire = None
            return Action.REFILL
        else:
            return self._move_towards((r, c), obs.actions)


    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    name = "H1"

    """
    H1 will always go to the closest fire and extinguish it (if it runs out of water,
    it will go to the closest water source to refill)
    """

    def __init__(self, player):
        super().__init__(player)
    
    def step(self, obs):

        if self.water == 0:
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if self.target_fire is not None and obs.field[self.target_fire[0], self.target_fire[1]] != 0:
                r,c = self.target_fire
            else:
                res = self._closest_fire(obs)
                if res is None:
                    return Action.NONE
                    
                r,c = res
                self.target_fire = (r,c)

            dist = (abs(r - y) + abs(c - x))
            if dist == 0 or dist == 1:
                self.target_fire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)


class H2(HeuristicAgent):
    name = "H2"

    """
    H2 will always go to the strongest front of the closest fire and extinguish it 
    (if it runs out of water, it will go to the closest water source to refill)
    """

    def __init__(self, player):
        super().__init__(player)
        self.player.extinguishing_mode = ExtinguishingMode.STRONGEST

    def step(self, obs):
        if self.water == 0:
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if self.target_fire is not None and obs.field[self.target_fire[0], self.target_fire[1]] != 0:
                r,c = self.target_fire
            else:
                res = self._strongest_front_closest_fire(obs)
                if res is None:
                    return Action.NONE
                    
                r,c = res
                self.target_fire = (r,c)

            dist = (abs(r - y) + abs(c - x))
            if dist == 0 or dist == 1:
                self.target_fire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)


class H3(HeuristicAgent):
    name = "H3"

    """
    H3 Agent always goes to the closest fire that it can put out or will go and refill water.
    If it can't put out any more fires, it won't do anything else
    """

    def __init__(self, player):
        super().__init__(player)

    def step(self, obs):

        if self.water == 0:
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if self.target_fire is not None and obs.field[self.target_fire[0], self.target_fire[1]] != 0:
                r,c = self.target_fire
            else:
                res = self._closest_fire(obs, max_fire_level=self.water//100)
                if res is None:
                    if self.water == self.controller.water_capacity: # no more fires can be extinguished by this agent
                        return Action.NONE
                    else:
                        return self._refill_water(obs)
               
                r,c = res
                self.target_fire = (r,c)

            dist = (abs(r - y) + abs(c - x))
            if dist == 0 or dist == 1:
                self.target_fire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)
                

class H4(HeuristicAgent):
    name = "H4"

    """
    H4 Agent always goes to the strongest fire burning (at the moment) or will go and refill water.
    """

    def __init__(self, player):
        super().__init__(player)
        self.player.extinguishing_mode = ExtinguishingMode.STRONGEST

    def step(self, obs):

        if self.water == 0:
            self.target_fire = None
            return self._refill_water(obs)
        else:
            y, x = self.observed_position
            if self.target_fire is not None and obs.field[self.target_fire[0], self.target_fire[1]] != 0:
                r,c = self.target_fire
            else:
                r,c = np.unravel_index(np.argmax(obs.field), obs.field.shape)
                self.target_fire = (r,c)

            dis = abs(r - y) + abs(c - x)
            if dis == 0 or dis == 1:
                self.target_fire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)
            
class H5(HeuristicAgent):
    name = "H5"

    """
    H5 Agent always goes to the weakest fire burnning (at the moment) or will go and refill water.
    """

    def __init__(self, player):
        super().__init__(player)
        self.player.extinguishing_mode = ExtinguishingMode.WEAKEST

    def step(self, obs):

        if self.water == 0:
            self.target_fire = None
            return self._refill_water(obs)
        else:
            y, x = self.observed_position
            if self.target_fire is not None and obs.field[self.target_fire[0], self.target_fire[1]] != 0:
                r,c = self.target_fire
            else:
                field = np.copy(obs.field)
                field[field == -1] = np.iinfo(np.int32).max
                field[field == 0] = np.iinfo(np.int32).max
                r,c = np.unravel_index(np.argmin(field), field.shape)
                self.target_fire = (r,c)

            dis = abs(r - y) + abs(c - x)
            if dis == 0 or dis == 1:
                self.target_fire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)