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
        self.old_target = None
        self.old_assigned_fire = None
        self.conventions = None

    def social_conventions(self, obs):
        raise NotImplemented("Conventions are implemented in the subclasse")
    
    def check_fire_extinguished(self, tiles, fires):
        all_fires_tiles = [[tile.row, tile.col] for fire in fires for tile in fire]
        return all([tile not in all_fires_tiles for tile in tiles])
    
    def check_target_extinguished(self, tile, fires):
        all_fires_tiles = [[tile.row, tile.col] for fire in fires for tile in fire]
        return tile not in all_fires_tiles
    
    def get_agent_index_in_convention(self, agents_convention):
        for index, agent in agents_convention.items():
            if (agent.is_self):
                return index
    
    def assign_fire_to_agent(self, fires_convention, agents_convention):
        agent_index = self.get_agent_index_in_convention(agents_convention)

        if (len(fires_convention) >= len(agents_convention)):
            agent_order = agent_index
        else:
            agent_order = agent_index % len(fires_convention)

        assigned_fire = fires_convention[agent_order]  
        self.old_assigned_fire = assigned_fire
        self.old_target = assigned_fire[0] 


    def step(self, obs):

        # compute conventions if not computed yet
        if (self.conventions == None):
            self.conventions = self.social_conventions(obs)

        # get agents and fires conventions
        agents_convention, fires_convention = self.conventions[0], self.conventions[1]

        # update fires_convention to remove possible extinguished fires; remove empty fires
        for fire in fires_convention:
            for tile in fire:
                if ([tile.row, tile.col] not in [[obs_tile.row, obs_tile.col] for obs_fire in obs.fires for obs_tile in obs_fire]):
                    fire.remove(tile)
        fires_convention = [fire for fire in fires_convention if fire]

        # if water level is zero, refill
        if(self.water == 0):
            return self._refill_water(obs)  
        
        if (self.old_target != None and self.old_assigned_fire != None):
            # if assigned fire has not been extinguished
            old_assigned_fires_tiles_fields = [[tile.row, tile.col] for tile in self.old_assigned_fire]
            if (not self.check_fire_extinguished(old_assigned_fires_tiles_fields, obs.fires)):   

                # if tile target was extinguished then change target to another tile of the same fire
                old_target_tile = [self.old_target.row, self.old_target.col]
                if (self.check_target_extinguished(old_target_tile, obs.fires)):
                    self.old_target = self.old_assigned_fire[0]

            else:
                self.assign_fire_to_agent(fires_convention, agents_convention)
        else: 
            self.assign_fire_to_agent(fires_convention, agents_convention)    
           
        # if agent is close to fire extinguishes it
        y,x = self.observed_position
        frow, fcol = self.old_target.row, self.old_target.col 
        if abs(frow - y) + abs(fcol - x) == 1:
            return Action.EXTINGUISH 
            
        return self._move_towards((self.old_target.row, self.old_target.col), obs.actions)     


class C1(ConventionAgent):
    name = "C1"

    """
    The agents and fires are in descending order of level, so agents with greater level go to fires with greater level
    If there are more fires than agents, each agent get the fires with id = multiple of the agent's id
    If there are more agents than fires, each agent get the fire with id = len(fires) % agent_id
    """

    def social_conventions(self, obs):

        # sort fires by descending order of the sum of tiles level
        fire_indices = np.arange(len(obs.fires))
        fire_levels_by_indice = dict(zip(fire_indices, (sum([tile.level for tile in fire]) for fire in obs.fires)))
        fire_indices_sorted_by_level = dict(sorted(fire_levels_by_indice.items(), key=lambda x: x[1], reverse=True))
        fire_convention_grouped = [obs.fires[indice] for indice in fire_indices_sorted_by_level]

        # sort agent by descending level
        agent_ids = np.arange(len(obs.players))
        agent_sorted = sorted(obs.players, key=lambda x: x.level, reverse=True)
        agent_convention = dict(zip(agent_ids, agent_sorted))

        return [agent_convention, fire_convention_grouped]


class C2(ConventionAgent):
    name = "C2"

    """
    The agents and fires are ordered by id 
    """

    def social_conventions(self, obs):
        fire_convention = [fire for fire in obs.fires ]

        agent_ids = np.arange(len(obs.players))
        agent_convention = dict(zip(agent_ids, obs.players))

        return agent_convention, fire_convention
    

class C3(ConventionAgent):
    name = "C3"

    """
    The agents are order by descending level and fires are ordered by ascending level
    """

    def social_conventions(self, obs):
        # sort fires by ascending order of the sum of tiles level
        fire_indices = np.arange(len(obs.fires))
        fire_levels_by_indice = dict(zip(fire_indices, (sum([tile.level for tile in fire]) for fire in obs.fires)))
        fire_indices_sorted_by_level = dict(sorted(fire_levels_by_indice.items(), key=lambda x: x[1]))
        fire_convention_grouped = [obs.fires[indice] for indice in fire_indices_sorted_by_level]

        # sort agent by descending level
        agent_ids = np.arange(len(obs.players))
        agent_sorted = sorted(obs.players, key=lambda x: x.level, reverse=True)
        agent_convention = dict(zip(agent_ids, agent_sorted))
        
        return [agent_convention, fire_convention_grouped]  
      
    
class C4(ConventionAgent):
    name = "C4"

    """
    The agents and fires are ordered by descending order of level, but the social convention is recalculated at each step
    """

    def social_conventions(self, obs):
        # sort fires by descending order of the sum of tiles level
        fire_indices = np.arange(len(obs.fires))
        fire_levels_by_indice = dict(zip(fire_indices, (sum([tile.level for tile in fire]) for fire in obs.fires)))
        fire_indices_sorted_by_level = dict(sorted(fire_levels_by_indice.items(), key=lambda x: x[1], reverse=True))
        fire_convention_grouped = [obs.fires[indice] for indice in fire_indices_sorted_by_level]

        # sort agent by descending level
        agent_ids = np.arange(len(obs.players))
        agent_sorted = sorted(obs.players, key=lambda x: x.level, reverse=True)
        agent_convention = dict(zip(agent_ids, agent_sorted))

        return [agent_convention, fire_convention_grouped]
    
    def step(self, obs):
        # get conventions
        agents_convention, fires_convention = self.social_conventions(obs)

        if(self.water == 0):
            return self._refill_water(obs)

        # get agent's assigned fire
        agent_index = self.get_agent_index_in_convention(agents_convention)
        if (len(fires_convention) >= len(agents_convention)):
            agent_order = agent_index
        else:
            agent_order = agent_index % len(fires_convention)     
        assigned_fire = fires_convention[agent_order]  

        # if agent is close to fire extinguishes it
        y,x = self.observed_position
        frow, fcol = assigned_fire[0].row, assigned_fire[0].col 
        if abs(frow - y) + abs(fcol - x) == 1:
            return Action.EXTINGUISH                
           
        return self._move_towards((frow, fcol), obs.actions)  