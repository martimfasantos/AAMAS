import random
import numpy as np
from . import Agent
from ..foraging.environment import Action,ExtiguishingMode
from ..foraging import FireTruck


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def __init__(self, player):
        super().__init__(player)

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))
    
    def _compute_turn(self, direction, targetedDirection):
        turns = [Action.NONE, Action.TURN_RIGHT, Action.TURN_AROUND, Action.TURN_LEFT]
        return turns[(targetedDirection.value - direction.value) % 4]

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        # self.forbidden prevents the agent from going back and forth between two tiles
        if r < y and Action.NORTH:
            if Action.NORTH in allowed:
                return Action.NORTH
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.NORTH)
        
        if r > y and Action.SOUTH:
            if Action.SOUTH in allowed:
                return Action.SOUTH
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.SOUTH)
            
        
        if c > x and Action.EAST:
            if(Action.EAST in allowed):
                return Action.EAST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.EAST)
            
        
        if c < x and Action.WEST:
            if(Action.WEST in allowed):
                return Action.WEST
            elif(isinstance(self.controller, FireTruck)):
                return self._compute_turn(self.direction, Action.WEST)
            
        return Action.NONE
            

    def _refill_water(self, obs):
        
        y,x = self.observed_position
        if(self.targetWaterResource is not None):
            r,c = self.targetWaterResource
        else:
            r,c = self._closest_water_source(obs)
            self.targetFire = (r,c)


        if (abs(r - y) + abs(c - x)) == 1:
            self.targetFire = None
            return Action.REFILL
        else:
            return self._move_towards((r, c), obs.actions)


    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
     H1 will always go to the closest fire and extinguish it (if it runs out of water,
       it will go to the closest water source to refill)
     """


    name = "H1"

    def __init__(self, player):
        super().__init__(player)
        self.targetFire = None
        self.targetWaterResource = None

    
    def step(self, obs):

        if(self.water == 0):
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if(self.targetFire is not None and obs.field[*self.targetFire] != 0):
                r,c = self.targetFire
            else:
                res = self._closest_fire(obs)
                if(res is None ):
                    return Action.NONE
                    
               
                r,c = res
                self.targetFire = (r,c)


            if (abs(r - y) + abs(c - x)) == 1:
                self.targetFire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)

# TODO: Adapt this agent   
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
     H2 Agent always goes to the closest fire that it can put out or will go and refill water.
     If it can't put out any more fires, it won't do anything else
     """


    name = "H2"

    def __init__(self, player):
        super().__init__(player)
        self.targetFire = None
        self.targetWaterResource = None



    def step(self, obs):

        if(self.water == 0):
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if(self.targetFire is not None and obs.field[*self.targetFire] != 0):
                r,c = self.targetFire
            else:
                res = self._closest_fire(obs,max_fire_level=self.water//100)
                if(res is None ):
                    if(self.water == self.controller.water_capacity): # no more firs can be extinguished by this agent
                        return Action.NONE
                    else:
                        return self._refill_water(obs)
               
                r,c = res
                self.targetFire = (r,c)


            if (abs(r - y) + abs(c - x)) == 1:
                self.targetFire = None
                return Action.EXTINGUISH
            else:
                return self._move_towards((r, c), obs.actions)
                

class H3(HeuristicAgent):
    name = "H3"

    """
     H3 Agent always goes to the strongest fire burnning (at the moment) or will go and refill water.
     """

    def __init__(self, player):
        super().__init__(player)
        self.targetFire = None
        self.targetWaterResource = None
        self.player.extiguishingMode = ExtiguishingMode.STRONGEST



    def step(self, obs):

        if(self.water == 0):
            self.targetFire = None
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if(self.targetFire is not None and obs.field[*self.targetFire] != 0):
                r,c = self.targetFire
            else:
                r,c = np.unravel_index(np.argmax(obs.field), obs.field.shape)
                self.targetFire = (r,c)

            dis = abs(r - y) + abs(c - x)
            if dis == 1:
                self.targetFire = None
                return Action.EXTINGUISH
            elif dis == 0:
                return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
            else:
                return self._move_towards((r, c), obs.actions)
            
class H4(HeuristicAgent):
    name = "H4"

    """
     H3 Agent always goes to the weakest fire burnning (at the moment) or will go and refill water.
     """

    def __init__(self, player):
        super().__init__(player)
        self.targetFire = None
        self.targetWaterResource = None
        self.player.extiguishingMode = ExtiguishingMode.WEAKEST



    def step(self, obs):

        if(self.water == 0):
            self.targetFire = None
            return self._refill_water(obs)
        else:
            y,x = self.observed_position
            if(self.targetFire is not None and obs.field[*self.targetFire] != 0):
                r,c = self.targetFire
            else:
                field = np.copy(obs.field)
                field[field == -1] = np.iinfo(np.int32).max
                field[field == 0] = np.iinfo(np.int32).max
                r,c = np.unravel_index(np.argmin(field), field.shape)
                self.targetFire = (r,c)

            dis = abs(r - y) + abs(c - x)
            if dis == 1:
                self.targetFire = None
                return Action.EXTINGUISH
            elif dis == 0:
                return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
            else:
                return self._move_towards((r, c), obs.actions)