import numpy as np
from .heuristic_agent import HeuristicAgent
from ..foraging.environment import Action

ROLE_ASSIGNMENT_PERIOD = 3


class RoleAgent(HeuristicAgent):
    name = "Role Agent"

    def __init__(self, player):
        super().__init__(player)
        self.curr_role = None
        self.role_assignment_period = None
        self.role_assignments = None
        self.steps_counter = 0


    def potential_function(self, agent, fire):
        """
        Calculates the potential function used for role assignment.
        """
        raise NotImplemented("Potential Functions are implemented in the subclasses")

    def role_assignment(self, obs):

        potentials = np.zeros((len(obs.fires), len(obs.players)))
        fire_levels = [sum([fire.level for fire in fires]) for fires in obs.fires]

        for i, fire in enumerate(obs.fires):
            for j, agent in enumerate(obs.players):
                potentials[i, j] = self.potential_function(agent, fire)

        agents_roles = np.zeros(len(obs.players), dtype=np.int32)
        assigned_levels = np.zeros(len(obs.fires), dtype=np.int32)

        # sort agent by level in descending order to improve efficiency
        agent_ids = np.arange(len(obs.players))
        agent_list = list(zip(agent_ids, obs.players))
        sorted_agents = sorted(agent_list, key=lambda x: x[1].level, reverse=True)

        # Assign roles based on max potential values for each agents
        for id, agent in sorted_agents:
            fire_id = np.argmax(potentials[:, id])
            agents_roles[id] = fire_id
            assigned_levels[fire_id] += obs.players[id].level
        
        # get the fire with the worst potential values
        worst_potential_fire_id = np.argmin(sum(potentials[fire_id, :]) 
                                            for fire_id in range(len(obs.fires)))

        # Iterate through the fires to find those that do not 
        # require as much level/water as they have been assigned
        over_watered_fires = np.where(assigned_levels > fire_levels)[0]

        while len(over_watered_fires) > 0:
            fire_id = over_watered_fires[0]
            over_watered_fires = np.delete(over_watered_fires, 0)
            # fire is already ready to be extinguised
            potentials[fire_id, :] = -999
            # sort the agents by their potential values (highest first)
            agents_assigned = sorted(np.where(agents_roles == fire_id)[0],
                                    key=lambda x: potentials[fire_id, x], reverse=True)
            
            for agent_id in self._find_unecessary_agents(obs, agents_assigned, fire_levels[fire_id]):
                new_fire_id = np.argmax(potentials[:, agent_id])
                # if all other fires are ready to be extinguished, keep the same
                if potentials[new_fire_id, agent_id] == -999:
                    new_fire_id = worst_potential_fire_id
                # else, assign agent other fire based
                else:
                    assigned_levels[new_fire_id] += obs.players[agent_id].level
                    if assigned_levels[new_fire_id] > fire_levels[new_fire_id]:
                        over_watered_fires = np.append(over_watered_fires, new_fire_id)

                agents_roles[agent_id] = new_fire_id

        return agents_roles

    
    def _find_unecessary_agents(self, obs, agents_assigned, fire_level):
        """
        Finds the agents that are not needed for a fire.
        """
        agents_level = 0
        i = 0
        for i, agent_id in enumerate(agents_assigned):
            if agents_level < fire_level:
                agents_level += obs.players[agent_id].level
            else:
                break
        return agents_assigned[i:]
    
    def _better_to_refill(self, obs, fire_pos, distance, fire_level):
        """
        Checks if it is better to refill water.
        """
        dist_water_source = float('inf')
        water_source = None

        for ws in obs.water_sources:
            frow, fcol = ws.row, ws.col
            dist = self.manhattan_distance((frow, fcol), self.position)            
            if dist < dist_water_source:
                dist_water_source = dist
                water_source = (frow, fcol)
        
        dist_water_source_to_fire = self.manhattan_distance(water_source, fire_pos)
        if self.water < fire_level and self.water_capacity >= fire_level:
            return distance + 2 * dist_water_source_to_fire > dist_water_source + distance
        
        return self.water/distance < self.water_capacity/(dist_water_source + dist_water_source_to_fire)
        
    
    @staticmethod
    def manhattan_distance(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
        

    def step(self, obs):

        # set the role assignment period
        if self.role_assignment_period is None:
            self.role_assignment_period = ROLE_ASSIGNMENT_PERIOD * len(obs.players)

        self.steps_counter += 1

        # reassign roles if it is time to do so
        if self.steps_counter % self.role_assignment_period == 0 \
            or self.role_assignments is None:
            self.role_assignments = self.role_assignment(obs)

        self.curr_role = self.role_assignments[self.id]
            
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

        min_distance = float('inf')
        target = None

        for fire in assigned_fire:
            frow, fcol = fire.row, fire.col
            dist = self.manhattan_distance((frow, fcol), (y, x))
            if dist == 0 or dist == 1:
                return Action.EXTINGUISH
            
            if dist < min_distance:
                min_distance = dist
                target = (frow, fcol)
        
        if self._better_to_refill(obs, target, min_distance, sum(fire.level for fire in assigned_fire)):
            return self._refill_water(obs)

        return self._move_towards(target, obs.actions)

class R1(RoleAgent):
    name = "R1"

    """
    The fire's roles prioritize the closest agents with the most remaining water
    """

    def potential_function(self, agent, fire):
        return self.water \
            - min([self.manhattan_distance(agent.position, (tile.row, tile.col)) for tile in fire])


class R2(RoleAgent):
    name = "R2"

    """
    The fire's roles are assigned based on the distance to the closest fire tile
    """

    def potential_function(self, agent, fire):
        return -min([self.manhattan_distance(agent.position, (tile.row, tile.col)) for tile in fire])
    

class R3(RoleAgent):
    name = "R3"

    """
    The fire's roles prioritize the closest agents with the most percentage of water
    compared to its total capacity
    """

    def potential_function(self, agent, fire):
        return self.water / self.water_capacity \
            - min([self.manhattan_distance(agent.position, (tile.row, tile.col)) for tile in fire])
    