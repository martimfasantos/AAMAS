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

        print([[x[0], x[1].controller.water] for x in agents_convention.items()])
        print(fires_convention)
        print("\n")

        if(self.water == 0 or agent_id not in agents_convention):
            return self._refill_water(obs)

        # get agent's assigned fire
        agent_order = list(agents_convention).index(self.id)
        
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
    name = "C2"

    def social_conventions(self, obs):
        fire_convention = [tile for fire in obs.fires for tile in fire]

        agent_ids = np.arange(len(obs.players))
        agent_convention = dict(zip(agent_ids, obs.players))

        return agent_convention, fire_convention
    

class C3(ConventionAgent):
    """
    The agents are order by level and use tiles ordered by level instead of fires
    """
    name = "C3"

    def social_conventions(self, obs):
        tiles = [tile for fire in obs.fires for tile in fire]
        fire_convention = sorted(tiles, key=lambda x: x.level, reverse=True)

        # sort agent by level
        agent_ids = np.arange(len(obs.players))
        agent_dict = dict(zip(agent_ids, obs.players))
        agent_convention = dict(sorted(agent_dict.items(), key=lambda x: x[1].level, reverse=True))

        print(agent_convention)
        print(fire_convention)
        
        return agent_convention, fire_convention
    
class C4(ConventionAgent):
    """
    The agents are order by level and use tiles ordered by ascending level instead of fires
    """
    name = "C4"

    def social_conventions(self, obs):
        tiles = [tile for fire in obs.fires for tile in fire]
        fire_convention = sorted(tiles, key=lambda x: x.level)

        # sort agent by level
        agent_ids = np.arange(len(obs.players))
        agent_dict = dict(zip(agent_ids, obs.players))
        agent_convention = dict(sorted(agent_dict.items(), key=lambda x: x[1].level, reverse=True))
        
        return agent_convention, fire_convention
    
    
class C5(ConventionAgent):
    """
    The agents are order by id and use tiles ordered by ascending level instead of fires
    """
    name = "C5"

    def social_conventions(self, obs):
        tiles = [tile for fire in obs.fires for tile in fire]
        fire_convention = sorted(tiles, key=lambda x: x.level)

        # sort agent by id
        agent_ids = np.arange(len(obs.players))
        agent_convention = dict(zip(agent_ids, obs.players))
        
        return agent_convention, fire_convention
    
class C6(ConventionAgent):
    """
    Same as C5 but agents with no water don't have a tile assigned
    """
    name = "C6"

    def social_conventions(self, obs):

        def filter_agents_with_no_water(pair):
            id, agent = pair
            return agent.level > 0

        tiles = [tile for fire in obs.fires for tile in fire]
        fire_convention = sorted(tiles, key=lambda x: x.level)

        # sort agent by id
        agent_ids = np.arange(len(obs.players))
        agent_dict = dict(zip(agent_ids, obs.players))
        agent_convention = dict(filter(filter_agents_with_no_water, agent_dict.items()))
        
        return agent_convention, fire_convention