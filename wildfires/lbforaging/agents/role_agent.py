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

        agents_roles = np.zeros(len(obs.players),dtype=np.int32)
        assigned_water = np.zeros(len(obs.fires), dtype=np.int32)

        # Assign roles based on potential values and available water
        for id, agent in enumerate(obs.players):
            fire_id = np.argmax(potentials[:, id])
            agents_roles[id] = fire_id
            assigned_water[fire_id] += obs.players[id].water_available // 100 # safeguards from the case where the agent has no water
        
        # get the fires that were assigned more water than needed
        over_watered_fires = np.where(assigned_water > total_fire_levels)[0]
        fire_idx = 0
        while len(over_watered_fires) > 0:
            fire_id = over_watered_fires[0]
            over_watered_fires = np.delete(over_watered_fires, 0)
            agents_assigned = np.where(agents_roles == fire_id)[0]
            # sort the agents by their potential values (highest first)
            agents_assigned = sorted(agents_assigned, key=lambda x: potentials[fire_id, x], reverse=True)
            # find the agents that aren't needed for this fire (by summing their remaining water)
            potentials[fire_id, :] = -999

            for agent_id in self._find_unecessary_agents(obs, agents_assigned, total_fire_levels[fire_id]):
                new_fire_id = np.argmax(potentials[:, agent_id])
                if(potentials[new_fire_id, agent_id] == -999):
                    # if all fires have sufficient water, evenly distribute the remaining agents
                    new_fire_id = fire_idx
                    fire_idx = (fire_idx + 1) % len(obs.fires)
                else: 
                    assigned_water[new_fire_id] += obs.players[agent_id].water_available // 100
                    if(assigned_water[new_fire_id] > total_fire_levels[new_fire_id]):
                        over_watered_fires = np.append(over_watered_fires, new_fire_id)

                agents_roles[agent_id] = new_fire_id

                
        return agents_roles
    
    def _find_unecessary_agents(self, obs, agents_assigned, max_fire_level):
        """
        Finds the agents that are not needed for a fire.
        """
        aditional_agents = []
        water = 0
        i = 0
        for agent_id in agents_assigned:
            if(water < max_fire_level):
                aditional_agents.append(agent_id)
                water += obs.players[agent_id].water_available // 100
                i += 1
            else:
                break
        return agents_assigned[i:]
        
    
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
                assigned_fire = obs.fires[self.curr_role]
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
       


