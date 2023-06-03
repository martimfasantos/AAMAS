import numpy as np
from collections import namedtuple
from . import Agent
from ..foraging.environment import Action
from ..foraging import FireTruck
from .heuristic_agent import HeuristicAgent

class ConventionAgent(HeuristicAgent):
    name = "Social Convention agent"


    def __init__(self, player):
        super().__init__(player)

    def social_conventions(self, obs):
        raise NotImplemented("Conventions are implemented in the subclasse")

    def step(self, obs):
        # get conventions
        agents_convention, fires_convention = self.social_conventions(obs)

        if(self.water == 0):
            return self._refill_water(obs)

        # get agent's assigned fire
        agent_id = self.get_agent_id(agents_convention)
        agent_order = list(agents_convention).index(agent_id)

        if (agent_order >= len(fires_convention)):
            return Action.NONE
        else:
            assigned_fire = fires_convention[agent_order]  

            # if agent is close to fire extinguishes it
            y,x = self.observed_position
            frow, fcol = assigned_fire.row, assigned_fire.col 
            if abs(frow - y) + abs(fcol - x) == 1:
                return Action.EXTINGUISH                
               
            return self._move_towards((frow, fcol), obs.actions)

    def get_agent_id(self, agents_dict):
        for key,val in agents_dict.items():
            if (val.is_self):
                return key
        


class C1(ConventionAgent):
    """
    The agents and fires are in descending order of level, so agents with greater level go to fires with greater level
    """
    name = "C1"

    def social_conventions(self, obs):

        # sort fires by sum of tiles level
        fire_indices = np.arange(len(obs.fires))
        fire_levels_by_indice = dict(zip(fire_indices, (sum([tile.level for tile in fire]) for fire in obs.fires)))
        fire_indices_sorted_by_level = dict(sorted(fire_levels_by_indice.items(), key=lambda x: x[1], reverse=True))
        fire_convention_grouped = [obs.fires[indice] for indice in fire_indices_sorted_by_level]
        fire_convention_per_tile = [tile for fire in fire_convention_grouped for tile in fire]

        # sort agent by level
        agent_ids = np.arange(len(obs.players))
        agent_dict = dict(zip(agent_ids, obs.players))
        agent_convention = dict(sorted(agent_dict.items(), key=lambda x: x[1].level, reverse=True))

        return agent_convention, fire_convention_per_tile


class C2(ConventionAgent):
    """
    The agents and fires are ordered by id 
    """
    name = "C3"

    def social_conventions(self, obs):
        fire_convention = [tile for fire in obs.fires for tile in fire]

        agent_ids = np.arange(len(obs.players))
        agent_convention = dict(zip(agent_ids, obs.players))

        return agent_convention, fire_convention
    

class C3(ConventionAgent):
    """
    The agents are order by level and use tiles ordered by level instead of fires
    """
    name = "C4"

    def social_conventions(self, obs):
        tiles = [tile for fire in obs.fires for tile in fire]
        fire_convention = sorted(tiles, key=lambda x: x.level, reverse=True)

        # sort agent by level
        agent_ids = np.arange(len(obs.players))
        agent_dict = dict(zip(agent_ids, obs.players))
        agent_convention = dict(sorted(agent_dict.items(), key=lambda x: x[1].level, reverse=True))
        
        return agent_convention, fire_convention
    