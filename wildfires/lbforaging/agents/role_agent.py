import numpy as np
from itertools import chain
from .heuristic_agent import HeuristicAgent
from ..foraging.environment import Action
from ..foraging import FireTruck

class RoleAgent(HeuristicAgent):
    name = "Role Agent"

    def __init__(self, player):
        super().__init__(player)
        self.target_fire = None
        self.target_water_source = None
        self.role_assignment_period = 3 # TODO: review this OR make it a parameter
        self.role_assignments = None
        self.curr_role = None
        self.steps_counter = 0
        self.fire_levels = {}


    def potential_function(self, agent, fire):
        """
        Calculates the potential function used for role assignment.
        """
        raise NotImplemented("Potential Functions are implemented in the subclasses")

    def role_assignment(self, obs):

        potentials = np.zeros((len(obs.fires), len(obs.players)))
        total_fire_levels = [sum([fire.level for fire in fires]) for fires in obs.fires]

        for i, fire in enumerate(obs.fires):
            for j, agent in enumerate(obs.players):
                potentials[i, j] = self.potential_function(agent, fire)

        agents_roles = np.zeros(len(obs.players), dtype=np.int32)
        assigned_water = np.zeros(len(obs.fires), dtype=np.int32)

        # sort agent by level in descending order to improve efficiency
        agent_ids = np.arange(len(obs.players))
        agent_list = list(zip(agent_ids, obs.players))
        sorted_agents = sorted(agent_list, key=lambda x: x[1].level, reverse=True)

        # Assign roles based on potential values and available water
        for id, agent in sorted_agents:
            fire_id = np.argmax(potentials[:, id])
            agents_roles[id] = fire_id
            assigned_water[fire_id] += obs.players[id].level


            if assigned_water[fire_id] >= total_fire_levels[fire_id]:  # fire will be extinguished
                potentials[fire_id, :] = -999

        return agents_roles
    
    @staticmethod
    def manhattan_distance(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
        

    def step(self, obs):
        # assign the roles if it is time
        if self.curr_role is None or self.steps_counter % self.role_assignment_period == 0:
            self.role_assignments = self.role_assignment(obs)
            self.curr_role = self.role_assignments[self.id]
        
        self.steps_counter += 1

        if self.water == 0:
            return self._refill_water(obs)
        else:
            try:
                assigned_fire = obs.fires[self.role_assignments[self.id]]
            except IndexError:
                return Action.NONE

        if assigned_fire == []:
            return Action.NONE

        y, x = self.position

        minDistance = float('inf')
        target = (-1,-1)

        for fire in assigned_fire:
            frow, fcol = fire.row, fire.col
            dist = self.manhattan_distance((frow, fcol), (y, x))
            if dist == 0 or dist == 1:
                return Action.EXTINGUISH
            
            if dist < minDistance:
                minDistance = dist
                target = (frow, fcol)

        return self._move_towards(target, obs.actions)

class R1(RoleAgent):
    name = "R1"

    """
    The fire's roles prioritize the closest agents with the most remaining water
    """

    def potential_function(self, agent,fire):
        return self.controller.water \
            -min([self.manhattan_distance(agent.position,(tile.row,tile.col)) for tile in fire])

        

class R2(RoleAgent):
    name = "R2"

    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """

    def potential_function(self, agent, fire):
        return -min([self.manhattan_distance(agent.position,(tile.row,tile.col)) for tile in fire])
       


